"""
Implements the continuous fuzzy sets, using PyTorch.
"""

import torch
import numpy as np


class ContinuousFuzzySet(torch.nn.Module):
    """
    A generic and abstract torch.nn.Module class that implements continuous fuzzy sets.
    """
    def __init__(self, in_features, centers=None, widths=None, labels=None):
        super().__init__()
        self.in_features = in_features
        self._log_widths = None
        self.labels = labels

        # initialize centers
        if centers is None:
            self.centers = torch.nn.parameter.Parameter(torch.randn(self.in_features))
        else:
            centers = self.convert_to_tensor(centers)
            self.centers = torch.nn.parameter.Parameter(centers)

        # initialize widths -- never adjust the widths directly,
        # use the logarithm of them to avoid negatives
        if widths is None:  # we apply the logarithm to the widths,
            # so later, if we train them, and they become
            # nonzero, with an exponential function they are still positive
            # in other words, since gradient descent may make the widths negative,
            # we nullify that effect
            self.widths = torch.rand(self.in_features)
        else:
            # we assume the widths are given to us are within (0, 1)
            widths = self.convert_to_tensor(widths)
            self.widths = torch.nn.parameter.Parameter(torch.abs(widths))

        self.log_widths()  # update the stored log widths

    def log_widths(self):
        """
        Calculate the logarithm of the widths. Used for FLCs where the backpropagation may need
        to update the widths, and zero or negative values need to be avoided.

        Returns:
            The logarithm of the widths.
        """
        with torch.no_grad():
            self._log_widths = torch.nn.parameter.Parameter(torch.log(self.widths))
        return self._log_widths

    @staticmethod
    def convert_to_tensor(values):
        """
        If the given values are not torch.Tensor, convert them to torch.Tensor.

        Args:
            values: Values such as the centers or widths of a fuzzy set.

        Returns:
            torch.tensor(np.array(values)).float()
        """
        if isinstance(values, torch.Tensor):
            return values
        else:
            return torch.tensor(np.array(values)).float()

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
        self.log_widths()  # update the stored log widths

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
            if not isinstance(centers, torch.Tensor):
                centers = torch.tensor(np.array(centers))
            self.centers = torch.nn.Parameter(torch.cat([self.centers, centers]))
            if not isinstance(widths, torch.Tensor):
                widths = torch.tensor(widths)
            self.widths = torch.nn.Parameter(torch.cat([self.widths, widths]))
        self.log_widths()  # update the stored log widths

    def forward(self):
        """
        Calculate the membership of an element to this fuzzy set; not implemented as this is a
        generic and abstract class. This method is overridden by a class that specifies the type
        of fuzzy set (e.g., Gaussian, Triangular).

        Returns:
            None
        """
        raise NotImplementedError('The Base Fuzzy Set has no membership function defined.')


class Gaussian(ContinuousFuzzySet):
    """
    Implementation of the Gaussian membership function, written in PyTorch.
    """

    def __init__(self, in_features, centers=None, widths=None, labels=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - sigmas: trainable parameter
            centers and sigmas are initialized randomly by default,
            but sigmas must be > 0
        """
        super().__init__(in_features, centers, widths, labels)

    @property
    def sigmas(self):
        return self.widths

    @sigmas.setter
    def sigmas(self, sigmas):
        self.widths = sigmas

    def forward(self, observations):
        """
        Forward pass of the function. Applies the function to the input elementwise.

        Args:
            observations: Two-dimensional matrix of observations,
            where a row is a single observation and each column
            is related to an attribute measured during that observation.

        Returns:
            The membership degrees of the observations for the Gaussian fuzzy set.
        """
        return torch.exp(-1.0 * (torch.pow(
            self.convert_to_tensor(observations).unsqueeze(dim=-1) - self.centers,
            2) / torch.pow(torch.exp(self._log_widths), 2)))


class Triangular(ContinuousFuzzySet):
    """
    Implementation of the Triangular membership function, written in PyTorch.
    """

    def __init__(self, in_features, centers=None, widths=None, labels=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - widths: trainable parameter
            centers and widths are initialized randomly by default,
            but widths must be > 0
        """
        super().__init__(in_features, centers, widths, labels)

    def forward(self, observations):
        """
        Forward pass of the function. Applies the function to the input elementwise.

        Args:
            observations: Two-dimensional matrix of observations,
            where a row is a single observation and each column
            is related to an attribute measured during that observation.

        Returns:
            The membership degrees of the observations for the Triangular fuzzy set.
        """
        return torch.max(1.0 - (1.0 / torch.exp(
            self._log_widths)) * torch.abs(
            self.convert_to_tensor(
                observations).unsqueeze(dim=-1) - self.centers), torch.tensor(0.0))
