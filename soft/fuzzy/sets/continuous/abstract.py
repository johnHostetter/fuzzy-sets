"""
Implements an abstract class called ContinuousFuzzySet using PyTorch. All fuzzy sets defined over
a continuous domain are derived from this class. Further, the Membership class is defined within,
which contains a helpful interface understanding membership degrees.
"""
import inspect
from pathlib import Path
from collections import namedtuple
from abc import abstractmethod, ABC
from typing import List, NoReturn, Union, MutableMapping, Any, Set

import torch
import torchquad
import numpy as np

from soft.utilities.functions import convert_to_tensor, all_subclasses


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
        centers=None,
        widths=None,
        labels: List[str] = None,
    ):
        super().__init__()
        self.centers = torch.nn.Parameter(convert_to_tensor(centers).float())
        self.widths = torch.nn.Parameter(convert_to_tensor(widths).float())
        self.mask = self.widths > 0.0
        self.labels = labels

    @property
    def mask(self) -> torch.Tensor:
        """
        Get the mask of the fuzzy set, where the mask is a tensor of the same shape as the centers
        and widths. The mask has a value of 1 if the fuzzy set exists, and 0 if the fuzzy set does
        not exist.

        Returns:
            torch.Tensor
        """
        return self._mask

    @mask.setter
    def mask(self, mask) -> None:
        """
        Set the mask of the fuzzy set, where the mask is a tensor of the same shape as the centers
        and widths. The mask has a value of 1 if the fuzzy set exists, and 0 if the fuzzy set does
        not exist.

        Returns:
            torch.Tensor
        """
        # negative widths are a special flag to indicate that the fuzzy set
        # at that location does not actually exist
        self._mask = torch.Tensor(mask.int())  # keep only the valid fuzzy sets

    @mask.deleter
    def mask(self):
        """
        Delete the mask of the fuzzy set, where the mask is a tensor of the same shape as the
        centers and widths. The mask has a value of 1 if the fuzzy set exists, and 0 if the fuzzy
        set does not exist.

        Returns:
            torch.Tensor
        """
        del self._mask

    @classmethod
    def create(
        cls, number_of_variables: int, number_of_terms: int
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
        centers = torch.randn(number_of_variables, number_of_terms)
        widths = torch.rand(number_of_variables, number_of_terms)
        return cls(centers=centers, widths=widths)

    def __eq__(self, other: Any) -> bool:
        """
        Check if the fuzzy set is equal to another fuzzy set.

        Args:
            other: The other fuzzy set to compare to.

        Returns:
            True if the fuzzy sets are equal, False otherwise.
        """
        if not isinstance(other, type(self)):
            return False
        return torch.equal(self.centers, other.centers) and torch.equal(
            self.widths, other.widths
        )

    def __hash__(self):
        """
        Hash the fuzzy set.

        Returns:
            The hash of the fuzzy set.
        """
        return hash((type(self), self.centers, self.widths, self.labels))

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

    def reshape_parameters(self):
        """
        Reshape the parameters of the fuzzy set (e.g., centers, widths) so that they are
        the correct shape for subsequent operations.

        Returns:
            None
        """
        if self.centers.nelement() == 1:
            self.centers = torch.nn.Parameter(self.centers.reshape(1)).float()
        if self.widths.nelement() == 1:
            self.widths = torch.nn.Parameter(self.widths.reshape(1)).float()

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
            self.reshape_parameters()
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

            variables.append(
                type(self)(
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
            centers=centers.cpu().float().detach().tolist(),
            widths=widths.cpu().float().detach().tolist(),
        )

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
        observations: torch.Tensor = convert_to_tensor(observations).float()
        if observations.ndim <= self.centers.ndim:
            observations = observations.unsqueeze(dim=-1)
        # degrees = self.calculate_membership(observations)
        # mask = torch.zeros(degrees.shape[1:])
        # return Membership(degrees, mask)
        if observations.get_device() == -1:  # CPU
            degrees: torch.Tensor = self.cpu().calculate_membership(observations)
        else:  # GPU
            degrees: torch.Tensor = self.cuda().calculate_membership(observations)
        return Membership(degrees=degrees, mask=self.mask)
