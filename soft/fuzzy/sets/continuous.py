"""
Implements the continuous fuzzy sets, using PyTorch.
"""
from collections import namedtuple
from typing import List, NoReturn, Union, Tuple

import torch
import torchquad
import numpy as np

from utilities.functions import convert_to_tensor


class Membership(namedtuple(typename="Membership", field_names=("degrees", "mask"))):
    """
    The Membership class contains information describing both membership *degrees*
    and membership *mask*. The membership degrees are often the degree of membership, truth,
    activation, applicability, etc. of a fuzzy set, or more generally, a concept. The membership
    mask is shaped such that it helps filter or 'mask' out membership degrees that belong to
    fuzzy sets or concepts that are not actually real.

    The distinction between the two is made as applying the mask will zero out membership degrees
    that are not real, but this might be incorrectly interpreted as having zero degree of
    membership to the fuzzy set.
    """


class LogisticCurve(torch.nn.Module):
    """
    A generic torch.nn.Module class that implements a logistic curve, which allows us to
    tune the midpoint, and growth of the curve, with a fixed supremum (the supremum is
    the maximum value of the curve).
    """

    def __init__(self, midpoint, growth, supremum):
        super().__init__()
        self.midpoint = torch.nn.parameter.Parameter(
            convert_to_tensor(midpoint)
        ).float()
        self.growth = torch.nn.parameter.Parameter(
            convert_to_tensor(growth).double()
        ).float()
        self.supremum = convert_to_tensor(
            supremum  # not a parameter, so we don't want to track it
        ).float()

    def forward(self, input_data):
        """
        Calculate the value of the logistic curve at the given point.

        Args:
            input_data:

        Returns:

        """
        return self.supremum / (
            1 + torch.exp(-self.growth * (input_data - self.midpoint))
        )


class GroupedFuzzySets(torch.nn.Module):
    """
    A generic and abstract torch.nn.Module class that contains a torch.nn.ModuleList
    of ContinuousFuzzySet objects. The expectation here is that each ContinuousFuzzySet
    may define fuzzy sets of different conventions, such as Gaussian, Triangular, Trapezoidal, etc.
    Then, subsequent inference engines can handle these heterogeneously defined fuzzy sets
    with no difficulty. Further, this class was specifically designed to incorporate dynamic
    addition of new fuzzy sets in the construction of neuro-fuzzy networks via network morphism.

    However, this class does *not* carry out any functionality that is necessarily tied to fuzzy
    sets, it is simply named so as this was its intended purpose - grouping fuzzy sets. In other
    words, the same "trick" of using a torch.nn.ModuleList of torch.nn.Module objects applies to
    any kind of torch.nn.Module object.
    """

    def __init__(self, modules=None, expandable=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if modules is None:
            modules = []
        self.modules_list = torch.nn.ModuleList(modules)
        self.expandable = expandable
        self.epsilon = 0.5  # epsilon-completeness

    def __getattribute__(self, item):
        try:
            if item in ("centers", "widths", "sigmas"):
                modules_list = self.__dict__["_modules"]["modules_list"]
                if len(modules_list) > 0:
                    module_attributes: List[
                        torch.Tensor
                    ] = []  # the secondary response denoting module filter
                    for module in modules_list:
                        module_attributes.append(getattr(module, item))
                    return torch.cat(module_attributes, dim=-1)
                raise ValueError(
                    "The torch.nn.ModuleList of GroupedFuzzySets is empty."
                )
            return object.__getattribute__(self, item)
        except AttributeError:
            return self.__getattr__(item)

    def get_mask(self) -> Union[torch.Tensor, NoReturn]:
        if len(self.modules_list) > 0:
            module_masks: List[
                torch.Tensor
            ] = []  # the secondary response denoting module filter
            for module in self.modules_list:
                module_masks.append(module.get_mask().float())
            return torch.cat(module_masks, dim=-1)
        raise ValueError("The torch.nn.ModuleList of GroupedFuzzySets is empty.")

    def calculate_module_responses(
        self, observations
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], NoReturn]:
        if len(self.modules_list) > 0:
            # modules' responses are membership degrees when modules are ContinuousFuzzySet
            module_responses: List[
                torch.Tensor
            ] = []  # the primary response from the module
            module_masks: List[
                torch.Tensor
            ] = []  # the secondary response denoting module filter
            for module in self.modules_list:
                module_response, module_mask = module(observations)
                module_responses.append(module_response)
                module_masks.append(module_mask.float())
            return torch.cat(module_responses, dim=-1), torch.cat(module_masks, dim=-1)
        raise ValueError("The torch.nn.ModuleList of GroupedFuzzySets is empty.")

    def forward(self, observations) -> Membership:
        observations = convert_to_tensor(observations)
        module_responses, module_masks = self.calculate_module_responses(observations)

        # if mu.grad_fn is None and self.expandable:
        #     print("grad_fn of mu is None, but might be in target net tho")

        if self.expandable:
            # find where the new centers should be added, if any
            new_centers = (
                observations * (module_responses.max(dim=-1).values <= self.epsilon)
            ) + ((module_responses.max(dim=-1).values > self.epsilon) / 0).nan_to_num(
                0.0, posinf=torch.nan
            )
            # print(mu.max(dim=-1).values, observations)
            nc: List[torch.Tensor] = [
                torch.tensor(
                    list(
                        set(
                            new_centers[:, var_idx][
                                ~new_centers[:, var_idx].isnan()
                            ].tolist()
                        )
                    )
                )
                for var_idx in range(new_centers.shape[-1])
            ]

            # doing some trick w. dynamically expandable networks
            sets: List[ContinuousFuzzySet] = []
            empty_sets: List[int] = []
            for idx, n in enumerate(nc):
                if len(n) == 0:
                    empty_sets.append(idx)
                sets.append(
                    Gaussian(
                        in_features=len(n),
                        centers=n,
                        widths=torch.randn_like(n).abs(),
                    )
                )
            if len(sets) > 0 and len(empty_sets) < len(sets):
                # take the fuzzy sets, and make a ContinuousFuzzySet efficient object
                g = ContinuousFuzzySet.stack(sets)
                new_idx = len(self.modules_list)
                print(
                    f"adding {g.centers.shape}; num of modules already: {len(self.modules_list)}"
                )
                print(f"to dimensions: {set(range(len(sets))) - set(empty_sets)}")
                self.modules_list.add_module(str(new_idx), g)
                module_responses, module_masks = self.calculate_module_responses(
                    observations
                )

        return Membership(degrees=module_responses, mask=module_masks)


class ContinuousFuzzySet(torch.nn.Module):
    """
    A generic and abstract torch.nn.Module class that implements continuous fuzzy sets.

    This is the most important Python class regarding fuzzy sets within this Soft Computing library.

    Defined here are most of the common methods made available to all fuzzy sets. Fuzzy sets that
    will later be used in other features such as neuro-fuzzy networks are expected to abide by the
    conventions outlined within. For example, parameters 'centers' and 'widths' are often expected,
    but inference engines (should) only rely on the fuzzy set membership degrees.

    However, for convenience, some aspects of the SelfOrganize code may search for vertices that
    have attributes of type 'ContinuousFuzzySet'. Thus, if it is pertinent that a vertex within
    the KnowledgeBase is recognized as a fuzzy set, it is very likely one might be interested in
    inheriting or extending from ContinuousFuzzySet.
    """

    def __init__(self, in_features, centers=None, widths=None, labels=None):
        super().__init__()
        self.centers, self.widths, self.labels = None, None, None
        self.in_features, self.mask = None, None
        self.constructor(
            in_features=in_features, centers=centers, widths=widths, labels=labels
        )

    def constructor(self, in_features, centers=None, widths=None, labels=None):
        self.in_features = in_features
        self.labels = labels

        # initialize centers
        if centers is None:
            centers = torch.randn(self.in_features)
        else:
            centers = convert_to_tensor(centers)
        self.centers = torch.nn.parameter.Parameter(centers).float()

        # initialize widths -- never adjust the widths directly,
        # use the logarithm of them to avoid negatives
        if widths is None:  # we apply the logarithm to the widths,
            # so later, if we train them, and they become
            # nonzero, with an exponential function they are still positive
            # in other words, since gradient descent may make the widths negative,
            # we nullify that effect
            widths = torch.rand(self.in_features)
            mask = torch.ones(widths.shape)
        else:
            # we assume the widths are given to us are within (0, 1)
            widths = convert_to_tensor(widths)
            # negative widths are a special flag to indicate that the fuzzy set
            # at that location does not actually exist
            mask = (widths > 0).int()  # keep only the valid fuzzy sets

        self.widths = torch.nn.parameter.Parameter(widths).float()
        self.mask = mask

    def get_mask(self) -> torch.Tensor:
        # mask has value of 1 if you should ignore corresponding degree in same i'th and j'th place
        return (self.widths == -1.0).float()

    def reshape_parameters(self):
        """
        Reshape the parameters of the fuzzy set (e.g., centers, widths) so that they are
        the correct shape for subsequent operations.

        Returns:
            None
        """
        if self.centers.nelement() == 1:
            self.centers = torch.nn.Parameter(self.centers.reshape(1))
        if self.widths.nelement() == 1:
            self.widths = torch.nn.Parameter(self.widths.reshape(1))

    def extend(self, centers, widths):
        """
        Given additional parameters, centers and widths, extend the existing self.centers and
        self.widths, respectively. Additionally, update the necessary backend logic.

        Args:
            centers: The centers of new fuzzy sets.
            widths: The widths of new fuzzy sets.

        Returns:
            None
        """
        with torch.no_grad():
            self.in_features += len(centers)
            self.reshape_parameters()
            centers = convert_to_tensor(centers)
            self.centers = torch.nn.Parameter(torch.cat([self.centers, centers]))
            widths = convert_to_tensor(widths)
            self.widths = torch.nn.Parameter(torch.cat([self.widths, widths]))

    def area_helper(self, fuzzy_sets) -> List[float]:
        """
        Splits the fuzzy set (if representing a fuzzy variable) into individual fuzzy sets (the
        fuzzy variable's possible fuzzy terms), and does so recursively until the base case is
        reached. Once the base case is reached (i.e., a single fuzzy set), the area under its
        curve within the integration_domain is calculated. The result is a

        Args:
            fuzzy_sets: The fuzzy set to split into smaller fuzzy sets.

        Returns:
            A list of floats.
        """
        results = []
        for params in zip(fuzzy_sets.centers, fuzzy_sets.widths):
            centers, widths = params[0], params[1]
            fuzzy_set = self.__class__(
                in_features=centers.ndim, centers=centers, widths=widths
            )

            if centers.ndim > 0:
                results.append(self.area_helper(fuzzy_set))
            else:
                simpson_method = torchquad.Simpson()
                area = simpson_method.integrate(
                    fuzzy_set.calculate_membership,
                    dim=1,
                    N=101,
                    integration_domain=[
                        [
                            fuzzy_set.centers.item() - fuzzy_set.widths.item(),
                            fuzzy_set.centers.item() + fuzzy_set.widths.item(),
                        ]
                    ],
                )
                if fuzzy_set.widths.item() <= 0 and area != 0.0:
                    # if the width of a fuzzy set is negative or zero, it is a special flag that
                    # the fuzzy set does not exist; thus, the calculated area of a fuzzy set w/ a
                    # width <= 0 should be zero. However, in the case this does not occur,
                    # a zero will substitute to be sure that this issue does not affect results
                    area = 0.0
                results.append(area)

        return results

    def area(self) -> torch.Tensor:
        """
        Calculate the area beneath the fuzzy curve (i.e., membership function) using torchquad.

        This is a slightly expensive operation, but it is used for approximating the Mamdani fuzzy
        inference with arbitrary continuous fuzzy sets.

        Typically, the results will be cached somewhere, so that the area value can be reused.

        Returns:
            torch.Tensor
        """
        return torch.tensor(self.area_helper(self))

    def split(self) -> np.ndarray:
        """
        Efficient implementation of splitting *this* (self) fuzzy set, where each row contains the
        fuzzy sets for a fuzzy variable. For example, if we index the result with [0][1], then we
        would retrieve the second fuzzy (i.e., linguistic) term for the first fuzzy (i.e.,
        linguistic) variable.

        Returns:
            numpy.array
        """
        sets = []
        for center, width in zip(self.centers.flatten(), self.widths.flatten()):
            sets.append(Gaussian(1, center.item(), width.item()))
        return np.array(sets).reshape(self.centers.shape)

    def split_by_variables(self) -> List["ContinuousFuzzySet"]:
        """
        This operation takes the ContinuousFuzzySet and converts it to a list of ContinuousFuzzySet
        objects, if applicable. For example, rather than using a single Gaussian object to represent
        all Gaussian membership functions in the input space, this function will convert that to a
        list of Gaussian objects, where each Gaussian function is defined and restricted to a single
        input dimension. This is particularly helpful when modifying along a specific dimension.

        Returns:
            A list of ContinuousFuzzySet objects, where the length is equal to the number
            of input dimensions.
        """
        variables = []
        for centers, widths in zip(self.centers, self.widths):
            centers = centers.cpu().detach().tolist()
            widths = widths.cpu().detach().tolist()

            # the centers and widths must be trimmed to remove missing fuzzy set placeholders
            trimmed_centers, trimmed_widths = [], []
            for center, width in zip(centers, widths):
                if width > 0:
                    # if an input dimension has less fuzzy sets than another,
                    # then it is possible for the width entry to have '-1' as a
                    # placeholder indicating so
                    trimmed_centers.append(center)
                    trimmed_widths.append(width)

            in_features = len(trimmed_centers)
            variables.append(
                type(self)(
                    in_features=in_features,
                    centers=trimmed_centers,
                    widths=trimmed_widths,
                )
            )

        return variables

    @staticmethod
    def count_granule_terms(granules: List["ContinuousFuzzySet"]) -> np.ndarray:
        """
        Count the number of granules that occur in each dimension.

        Args:
            granules: A list of granules, where each granule is a ContinuousFuzzySet object.

        Returns:
            A Numpy array with shape (len(granules), ) and the data type is integer.
        """
        return np.array(
            [
                params.centers.size(dim=0) if params.centers.dim() > 0 else 0
                for params in granules
            ]
        ).astype("int32")

    @staticmethod
    def stack(
        granules: List["ContinuousFuzzySet"],
    ) -> "ContinuousFuzzySet":
        """
        Create a condensed and stacked representation of the given granules.

        Args:
            granules: A list of granules, where each granule is a ContinuousFuzzySet object.

        Returns:
            A ContinuousFuzzySet object.
        """
        if list(granules)[0].training:
            missing_center, missing_width = 0.0, -1.0
        else:
            missing_center = missing_width = torch.nan

        centers = torch.vstack(
            [
                torch.nn.functional.pad(
                    params.centers,
                    pad=(
                        0,
                        ContinuousFuzzySet.count_granule_terms(granules).max()
                        - params.centers.shape[0],
                    ),
                    mode="constant",
                    value=missing_center,
                )
                if params.centers.dim() > 0
                else torch.tensor(missing_center).repeat(
                    ContinuousFuzzySet.count_granule_terms(granules).max()
                )
                for params in granules
            ]
        )
        widths = torch.vstack(
            [
                torch.nn.functional.pad(
                    params.widths,
                    pad=(
                        0,
                        ContinuousFuzzySet.count_granule_terms(granules).max()
                        - params.widths.shape[0],
                    ),
                    mode="constant",
                    value=missing_width,
                )
                if params.centers.dim() > 0
                else torch.tensor(missing_center).repeat(
                    ContinuousFuzzySet.count_granule_terms(granules).max()
                )
                for params in granules
            ]
        )

        # prepare a condensed and stacked representation of the granules
        mf_type = type(granules[0])
        return mf_type(
            in_features=len(granules),
            centers=centers.cpu().detach().tolist(),
            widths=widths.cpu().detach().tolist(),
        )

    def forward(self, observations) -> NotImplementedError:
        """
        Calculate the membership of an element to this fuzzy set; not implemented as this is a
        generic and abstract class. This method is overridden by a class that specifies the type
        of fuzzy set (e.g., Gaussian, Triangular).

        Args:
            observations: Two-dimensional matrix of observations,
            where a row is a single observation and each column
            is related to an attribute measured during that observation.

        Returns:
            None
        """
        raise NotImplementedError(
            "The ContinuousFuzzySet has no defined membership function. Please create a class and "
            "inherit from ContinuousFuzzySet, or use a predefined class, such as Gaussian."
        )


class Gaussian(ContinuousFuzzySet):
    """
    Implementation of the Gaussian membership function, written in PyTorch.
    """

    def __init__(self, in_features, centers=None, widths=None, labels=None):
        super().__init__(in_features, centers=centers, widths=widths, labels=labels)

    @property
    def sigmas(self) -> torch.Tensor:
        """
        Gets the sigma for the Gaussian fuzzy set; alias for the 'widths' parameter.

        Returns:
            torch.Tensor
        """
        return self.widths

    @sigmas.setter
    def sigmas(self, sigmas) -> None:
        """
        Sets the sigma for the Gaussian fuzzy set; alias for the 'widths' parameter.

        Returns:
            None
        """
        self.widths = sigmas

    # def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
    #     # this is lorentzian
    #     return (1 - self.get_mask()).to(observations.device) * (
    #         1
    #         / (
    #             torch.pow(
    #                 (
    #                     self.centers.to(observations.device)
    #                     - observations.unsqueeze(dim=-1)
    #                 )
    #                 / (0.5 * self.widths.to(observations.device)),
    #                 2,
    #             )
    #             + 1
    #         )
    #     )

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        # this is Gaussian
        return (
            torch.exp(
                -1.0
                * (
                    torch.pow(
                        observations.unsqueeze(dim=-1) - self.centers,
                        2,
                    )
                    / (torch.pow(self.widths, 2) + 1e-32)
                )
            )
            * self.mask
        )

    def forward(self, observations) -> Membership:
        """
        Forward pass of the function. Applies the function to the input elementwise.

        Args:
            observations: Two-dimensional matrix of observations,
            where a row is a single observation and each column
            is related to an attribute measured during that observation.

        Returns:
            The membership degrees of the observations for the Gaussian fuzzy set.
        """
        observations = convert_to_tensor(observations)
        return Membership(
            degrees=self.calculate_membership(observations), mask=self.get_mask()
        )


class Triangular(ContinuousFuzzySet):
    """
    Implementation of the Triangular membership function, written in PyTorch.
    """

    def __init__(self, in_features, centers=None, widths=None, labels=None):
        super().__init__(in_features, centers=centers, widths=widths, labels=labels)

    def forward(self, observations) -> Membership:
        """
        Forward pass of the function. Applies the function to the input elementwise.

        Args:
            observations: Two-dimensional matrix of observations,
            where a row is a single observation and each column
            is related to an attribute measured during that observation.

        Returns:
            The membership degrees of the observations for the Triangular fuzzy set.
        """
        observations = convert_to_tensor(observations)

        return Membership(
            degrees=(
                torch.max(
                    1.0
                    - (1.0 / self.widths)
                    * torch.abs(observations.unsqueeze(dim=-1) - self.centers),
                    torch.tensor(0.0),
                )
                * self.mask
            ),
            mask=self.get_mask(),
        )
