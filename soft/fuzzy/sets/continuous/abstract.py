"""
Implements an abstract class called ContinuousFuzzySet using PyTorch. All fuzzy sets defined over
a continuous domain are derived from this class. Further, the Membership class is defined within,
which contains a helpful interface understanding membership degrees.
"""

import inspect
from pathlib import Path
from collections import namedtuple
from abc import abstractmethod, ABC
from typing import List, NoReturn, Union, MutableMapping, Any, Type, Tuple

import sympy
import torch
import torchquad
import numpy as np
import scienceplots
import matplotlib as mpl
import matplotlib.pyplot as plt

from soft.utilities.reproducibility import path_to_project_root
from soft.utilities.functions import convert_to_tensor, all_subclasses


class Membership(
    namedtuple(typename="Membership", field_names=("elements", "degrees", "mask"))
):
    """
    The Membership class contains information describing both membership *degrees* and
    membership *mask* for some given *elements*. The membership degrees are often the degree of
    membership, truth, activation, applicability, etc. of a fuzzy set, or more generally, a concept.
    The membership  mask is shaped such that it helps filter or 'mask' out membership degrees that
    belong to fuzzy sets or concepts that are not actually real.

    The distinction between the two is made as applying the mask will zero out membership degrees
    that are not real, but this might be incorrectly interpreted as having zero degree of
    membership to the fuzzy set.

    By including the elements' information with the membership degrees and mask, it is possible to
    keep track of the original elements that were used to calculate the membership degrees. This
    is useful for debugging purposes, and it is also useful for understanding the membership
    degrees and mask in the context of the original elements. Also, it can be used in conjunction
    with the mask to filter out membership degrees that are not real, as well as assist in
    performing advanced operations.
    """


class ContinuousFuzzySet(ABC, torch.nn.Module):
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

    def __init__(
        self,
        centers: np.ndarray,
        widths: np.ndarray,
        use_sparse_tensor=False,
        labels: List[str] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        # avoid allocating new memory for the centers and widths
        # use torch.float32 to save memory and speed up computations
        self.centers = torch.nn.Parameter(
            torch.as_tensor(centers, dtype=torch.float32, device=self.device),
            requires_grad=True,  # explicitly set to True
        )
        self.widths = torch.nn.Parameter(
            torch.as_tensor(widths, dtype=torch.float32, device=self.device),
            requires_grad=True,  # explicitly set to True
        )
        self.use_sparse_tensor = use_sparse_tensor
        self.mask = torch.nn.Parameter(
            torch.as_tensor(widths > 0.0, dtype=torch.int8, device=self.device),
            requires_grad=False,  # explicitly set to False (mask is not trainable)
        )
        self.labels = labels  # TODO: possibly remove this attribute

    @classmethod
    def create(
        cls, number_of_variables: int, number_of_terms: int, **kwargs
    ) -> Union[NoReturn, "ContinuousFuzzySet"]:
        """
        Create a fuzzy set with the given number of variables and terms, where each variable
        has the same number of terms. For example, if we have two variables, then we might have
        three terms for each variable, such as "low", "medium", and "high". This would result in
        a total of nine fuzzy sets. The centers and widths are initialized randomly.

        Args:
            number_of_variables: The number of variables.
            number_of_terms: The number of terms.

        Returns:
            A ContinuousFuzzySet object, or a NotImplementedError if the method is not implemented.
        """
        if inspect.isabstract(cls):
            # this error is thrown if the class is abstract, such as ContinuousFuzzySet, but
            # the method is not implemented (e.g., self.calculate_membership)
            raise NotImplementedError(
                "The ContinuousFuzzySet has no defined membership function. Please create a class "
                "and inherit from ContinuousFuzzySet, or use a predefined class, such as Gaussian."
            )
        centers: np.ndarray = np.random.randn(number_of_variables, number_of_terms)
        widths: np.ndarray = np.random.randn(number_of_variables, number_of_terms)
        return cls(centers=centers, widths=widths, **kwargs)

    def __eq__(self, other: Any) -> bool:
        """
        Check if the fuzzy set is equal to another fuzzy set.

        Args:
            other: The other fuzzy set to compare to.

        Returns:
            True if the fuzzy sets are equal, False otherwise.
        """
        return (
            isinstance(other, type(self))
            and torch.equal(self.centers, other.centers)
            and torch.equal(self.widths, other.widths)
        )

    def __hash__(self):
        """
        Hash the fuzzy set.

        Returns:
            The hash of the fuzzy set.
        """
        return hash((type(self), self.centers, self.widths, self.labels))

    @classmethod
    def render_formula(cls) -> sympy.Expr:
        """
        Render of the fuzzy set's membership function.

        Note: This is more beneficial for Python Console or Jupyter Notebook usage.

        Returns:
            Render of the fuzzy set's membership function.
        """
        sympy.init_printing(use_unicode=True)
        return cls.sympy_formula()

    @classmethod
    def latex_formula(cls) -> str:
        """
        String LaTeX representation of the fuzzy set's membership function.

        Note: This is more beneficial for animations or LaTeX documents.

        Returns:
            The LaTeX representation of the fuzzy set's membership function.
        """
        return sympy.latex(cls.sympy_formula())

    def save(self, path: Path):
        """
        Save the fuzzy set to a file.

        Returns:
            None
        """
        state_dict: MutableMapping = self.state_dict()
        state_dict["labels"] = self.labels
        state_dict["class_name"] = self.__class__.__name__
        if ".pt" not in path.name and ".pth" not in path.name:
            raise ValueError(
                f"The path to save the fuzzy set must have a file extension of '.pt', "
                f"but got {path.name}"
            )
        if ".pth" in path.name:
            raise ValueError(
                f"The path to save the fuzzy set must have a file extension of '.pt', "
                f"but got {path.name}. Please change the file extension to '.pt' as it is not "
                f"recommended to use '.pth' for PyTorch models, as it conflicts with Python path"
                f"configuration files."
            )
        torch.save(state_dict, path)
        return state_dict

    @staticmethod
    def get_subclass(class_name: str) -> Union[NoReturn, "ContinuousFuzzySet"]:
        """
        Get the subclass of ContinuousFuzzySet with the given class name.

        Args:
            class_name: The name of the subclass of ContinuousFuzzySet.

        Returns:
            A subclass of ContinuousFuzzySet.
        """
        fuzzy_set_class = None
        for subclass in all_subclasses(ContinuousFuzzySet):
            if subclass.__name__ == class_name:
                fuzzy_set_class = subclass
                break
        if fuzzy_set_class is None:
            raise ValueError(
                f"The fuzzy set class {class_name} was not found in the subclasses of "
                f"ContinuousFuzzySet. Please ensure that the fuzzy set class is a subclass of "
                f"ContinuousFuzzySet."
            )
        return fuzzy_set_class

    @classmethod
    def load(cls, path: Path) -> "ContinuousFuzzySet":
        """
        Load the fuzzy set from a file.

        Returns:
            None
        """
        state_dict: MutableMapping = torch.load(path)
        centers = state_dict.pop("centers")
        widths = state_dict.pop("widths")
        labels = state_dict.pop("labels")
        class_name = state_dict.pop("class_name")
        return cls.get_subclass(class_name)(
            centers=centers, widths=widths, labels=labels
        )

    # def reshape_parameters(self):
    #     """
    #     Reshape the parameters of the fuzzy set (e.g., centers, widths) so that they are
    #     the correct shape for subsequent operations.
    #
    #     Returns:
    #         None
    #     """
    #     if self.centers.nelement() == 1:
    #         self.centers = torch.nn.Parameter(self.centers.reshape(1)).float()
    #     if self.widths.nelement() == 1:
    #         self.widths = torch.nn.Parameter(self.widths.reshape(1)).float()

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
            # self.reshape_parameters()
            centers = convert_to_tensor(centers)
            self.centers = torch.nn.Parameter(
                torch.cat([self.centers, centers]).float()
            )
            widths = convert_to_tensor(widths)
            self.widths = torch.nn.Parameter(torch.cat([self.widths, widths]).float())

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
            fuzzy_set = self.__class__(centers=centers, widths=widths)

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
        return torch.tensor(self.area_helper(self)).float()

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
            sets.append(type(self)(1, center.item(), width.item()))
        return np.array(sets).reshape(self.centers.shape)

    def split_by_variables(self) -> Union[list, List[Type["ContinuousFuzzySet"]]]:
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

            variables.append(
                type(self)(
                    centers=np.array(trimmed_centers),
                    widths=np.array(trimmed_widths),
                )
            )

        return variables

    def plot(self, selected_terms: List[Tuple[int, int]] = None):
        """
        Plot the fuzzy set.

        Returns:
            None
        """
        if selected_terms is None:
            selected_terms = []

        with plt.style.context(["science", "no-latex", "high-contrast"]):
            for variable_idx in range(self.centers.shape[0]):
                _, ax = plt.subplots(1, figsize=(6, 4), dpi=100)
                mpl.rcParams["figure.figsize"] = (6, 4)
                mpl.rcParams["figure.dpi"] = 100
                mpl.rcParams["savefig.dpi"] = 100
                mpl.rcParams["font.size"] = 20
                mpl.rcParams["legend.fontsize"] = "medium"
                mpl.rcParams["figure.titlesize"] = "medium"
                mpl.rcParams["lines.linewidth"] = 2
                ax.tick_params(width=2, length=6)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                real_centers: List[float] = [
                    self.centers[variable_idx, term_idx].item()
                    for term_idx, mask_value in enumerate(self.mask[variable_idx])
                    if mask_value == 1
                ]
                real_widths: List[float] = [
                    self.widths[variable_idx, term_idx].item()
                    for term_idx, mask_value in enumerate(self.mask[variable_idx])
                    if mask_value == 1
                ]
                x_values = torch.linspace(
                    min(real_centers) - 2 * max(real_widths),
                    max(real_centers) + 2 * max(real_widths),
                    1000,
                )

                if self.centers.ndim == 1 or self.centers.shape[0] == 1:
                    x_values = x_values[:, None]
                elif self.centers.ndim == 2 or self.centers.shape[0] > 1:
                    x_values = x_values[:, None, None]

                memberships: torch.Tensor = self.calculate_membership(x_values)

                if memberships.ndim == 2:
                    memberships = memberships.unsqueeze(
                        dim=1
                    )  # add a temporary dimension for the variable

                memberships = memberships.detach().numpy()
                x_values = x_values.squeeze().detach().numpy()

                for term_idx in range(memberships.shape[-1]):
                    if self.mask[variable_idx, term_idx] == 0:
                        continue  # not a real fuzzy set
                    y_values = memberships[:, variable_idx, term_idx]
                    label: str = (
                        r"$\mu_{"
                        + str(variable_idx + 1)
                        + ","
                        + str(term_idx + 1)
                        + "}$"
                    )
                    if (variable_idx, term_idx) in selected_terms:
                        # edgecolor="#0bafa9"  # beautiful with facecolor=None  (AAMAS 2023)
                        plt.fill_between(
                            x_values, y_values, alpha=0.5, hatch="///", label=label
                        )
                    else:
                        plt.plot(x_values, y_values, alpha=0.5, label=label)
                plt.legend(
                    bbox_to_anchor=(0.5, -0.2),
                    loc="upper center",
                    ncol=len(real_centers),
                )
                plt.subplots_adjust(bottom=0.3, wspace=0.33)
                output_directory = path_to_project_root() / "output" / "figures"
                output_directory.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_directory / f"mu_{variable_idx}.png")
                plt.clf()

    @staticmethod
    def count_granule_terms(granules: List[Type["ContinuousFuzzySet"]]) -> np.ndarray:
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
            ],
            dtype=np.int8,
        )

    @staticmethod
    def stack(
        granules: List[Type["ContinuousFuzzySet"]],
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
                (
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
                )
                for params in granules
            ]
        )
        widths = torch.vstack(
            [
                (
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
                )
                for params in granules
            ]
        )

        # prepare a condensed and stacked representation of the granules
        mf_type = type(granules[0])
        return mf_type(
            centers=centers.cpu().detach().numpy(),
            widths=widths.cpu().detach().numpy(),
        )

    @classmethod
    @abstractmethod
    def sympy_formula(cls) -> sympy.Expr:
        """
        The abstract method that defines the membership function of the fuzzy set using sympy.

        Returns:
            A sympy.Expr object that represents the membership function of the fuzzy set.
        """
        raise NotImplementedError("The sympy_formula method must be implemented.")

    @abstractmethod
    def calculate_membership(
        self, observations: torch.Tensor
    ) -> Union[NoReturn, torch.Tensor]:
        """
        Calculate the membership of an element to this fuzzy set; not implemented as this is a
        generic and abstract class. This method is overridden by a class that specifies the type
        of fuzzy set (e.g., Gaussian, Triangular).

        Args:
            observations: Two-dimensional matrix of observations, where a row is a single
            observation and each column is related to an attribute measured during that observation.

        Returns:
            None
        """
        raise NotImplementedError(
            "The ContinuousFuzzySet has no defined membership function. Please create a class and "
            "inherit from ContinuousFuzzySet, or use a predefined class, such as Gaussian."
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
        if observations.ndim <= self.centers.ndim:
            observations = observations.unsqueeze(dim=-1)
        degrees: torch.Tensor = self.calculate_membership(observations)

        assert (
            not degrees.isnan().any()
        ), "NaN values detected in the membership degrees."
        assert (
            not degrees.isinf().any()
        ), "Infinite values detected in the membership degrees."

        # if observations.get_device() == -1:  # CPU
        #     degrees: torch.Tensor = self.cpu().calculate_membership(observations)
        # else:  # GPU
        #     degrees: torch.Tensor = self.cuda().calculate_membership(observations)
        return Membership(
            elements=observations,
            degrees=degrees.to_sparse() if self.use_sparse_tensor else degrees,
            mask=self.mask,
        )
