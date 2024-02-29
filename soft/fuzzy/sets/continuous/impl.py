"""
Implements various membership functions by inheriting from ContinuousFuzzySet.
"""

from typing import List

import torch

from soft.utilities.functions import convert_to_tensor
from soft.fuzzy.sets.continuous.abstract import ContinuousFuzzySet


class LogGaussian(ContinuousFuzzySet):
    """
    Implementation of the Log Gaussian membership function, written in PyTorch.
    This is a modified version that helps when the dimensionality is high,
    and TSK product inference engine will be used.
    """

    def __init__(
        self,
        centers=None,
        widths=None,
        width_multiplier: float = 1.0,  # in fuzzy logic, convention is usually 1.0, but can be 2.0
        labels: List[str] = None,
    ):
        super().__init__(centers=centers, widths=widths, labels=labels)
        self.width_multiplier = width_multiplier
        assert int(self.width_multiplier) in [1, 2]

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

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        return (
            -1.0
            * (
                torch.pow(
                    observations - self.centers,
                    2,
                )
                / (self.width_multiplier * torch.pow(self.widths, 2) + 1e-32)
            )
        ) * self.mask.to(observations.device)


class Gaussian(LogGaussian):
    """
    Implementation of the Gaussian membership function, written in PyTorch.
    """

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        return super().calculate_membership(observations).exp() * self.mask.to(
            observations.device
        )


class Lorentzian(ContinuousFuzzySet):
    """
    Implementation of the Lorentzian membership function, written in PyTorch.
    """

    def __init__(
        self,
        centers=None,
        widths=None,
        labels: List[str] = None,
    ):
        super().__init__(centers=centers, widths=widths, labels=labels)

    @property
    def sigmas(self) -> torch.Tensor:
        """
        Gets the sigma for the Lorentzian fuzzy set; alias for the 'widths' parameter.

        Returns:
            torch.Tensor
        """
        return self.widths

    @sigmas.setter
    def sigmas(self, sigmas) -> None:
        """
        Sets the sigma for the Lorentzian fuzzy set; alias for the 'widths' parameter.

        Returns:
            None
        """
        self.widths = sigmas

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mask.to(observations.device) * (
            1
            / (
                torch.pow(
                    (self.centers - observations) / (0.5 * self.widths),
                    2,
                )
                + 1
            )
        )


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

    def forward(self, tensors: torch.Tensor) -> torch.Tensor:
        """
        Calculate the value of the logistic curve at the given point.

        Args:
            tensors:

        Returns:

        """
        return self.supremum / (
            1 + torch.exp(-1.0 * self.growth * (tensors - self.midpoint))
        )


class Triangular(ContinuousFuzzySet):
    """
    Implementation of the Triangular membership function, written in PyTorch.
    """

    def __init__(
        self,
        centers=None,
        widths=None,
        labels: List[str] = None,
    ):
        super().__init__(centers=centers, widths=widths, labels=labels)

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the function. Applies the function to the input elementwise.

        Args:
            observations: Two-dimensional matrix of observations,
            where a row is a single observation and each column
            is related to an attribute measured during that observation.

        Returns:
            The membership degrees of the observations for the Triangular fuzzy set.
        """
        return torch.max(
            1.0 - (1.0 / self.widths) * torch.abs(observations - self.centers),
            torch.tensor(0.0),
        ) * self.mask.to(observations.device)
