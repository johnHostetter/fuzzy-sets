"""
Implements the continuous fuzzy sets, using PyTorch.
"""
import torch
import torchquad
import numpy as np

from utilities.functions import convert_to_tensor


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
            self.mask = torch.ones(widths.shape)
        else:
            # we assume the widths are given to us are within (0, 1)
            widths = convert_to_tensor(widths)
            # negative widths are a special flag to indicate that the fuzzy set
            # at that location does not actually exist
            self.mask = (widths > 0).int()  # keep only the valid fuzzy sets

        self.widths = torch.nn.parameter.Parameter(widths).float()
        self.mask = torch.nn.Parameter(
            self.mask.float(), requires_grad=False
        )  # mask is parameter, so it can easily switch from CPU to GPU
        self.log_widths()  # update the stored log widths

    def log_widths(self) -> torch.Tensor:
        """
        Calculate the logarithm of the widths. Used for FLCs where the backpropagation may need
        to update the widths, and zero or negative values need to be avoided.

        Returns:
            The logarithm of the widths.
        """
        with torch.no_grad():
            self._log_widths = torch.nn.parameter.Parameter(torch.exp(self.widths))
            if torch.isinf(self._log_widths).any().item():
                raise ValueError(
                    "Some of the widths are infinite, which is not allowed."
                )
        return self._log_widths

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
            centers = convert_to_tensor(centers)
            self.centers = torch.nn.Parameter(torch.cat([self.centers, centers]))
            widths = convert_to_tensor(widths)
            self.widths = torch.nn.Parameter(torch.cat([self.widths, widths]))
        self.log_widths()  # update the stored log widths

    def area_helper(self, fuzzy_sets):
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
                    fuzzy_set,
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

    def area(self):
        """
        Calculate the area beneath the fuzzy curve (i.e., membership function) using torchquad.

        This is a slightly expensive operation, but it is used for approximating the Mamdani fuzzy
        inference with arbitrary continuous fuzzy sets.

        Typically, the results will be cached somewhere, so that the area value can be reused.

        Returns:
            torch.Tensor
        """
        return torch.tensor(self.area_helper(self))

    def split(self):
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

    def forward(self, observations):
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
            "The Base Fuzzy Set has no membership function defined."
        )


class Gaussian(ContinuousFuzzySet):
    """
    Implementation of the Gaussian membership function, written in PyTorch.
    """

    @property
    def sigmas(self):
        """
        Gets the sigma for the Gaussian fuzzy set; alias for the 'widths' parameter.

        Returns:
            torch.Tensor
        """
        return self.widths

    @sigmas.setter
    def sigmas(self, sigmas):
        """
        Sets the sigma for the Gaussian fuzzy set; alias for the 'widths' parameter.

        Returns:
            None
        """
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
        observations = convert_to_tensor(observations)

        return (
            torch.exp(
                -1.0
                * (
                    torch.pow(
                        observations.unsqueeze(dim=-1) - self.centers, 2,
                    ) / (torch.pow(torch.log(self._log_widths.cuda()), 2) + 1e-32)
                )
            )
            * self.mask
        )


class Triangular(ContinuousFuzzySet):
    """
    Implementation of the Triangular membership function, written in PyTorch.
    """

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
        observations = convert_to_tensor(observations)

        return (
            torch.max(
                1.0
                - (1.0 / torch.log(self._log_widths))
                * torch.abs(
                    observations.unsqueeze(dim=-1) - self.centers
                ),
                torch.tensor(0.0),
            )
            * self.mask
        )
