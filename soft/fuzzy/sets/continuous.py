import torch
import numpy as np


class ContinuousFuzzySet(torch.nn.Module):
    def __init__(self, in_features, centers=None, widths=None, supports=None, labels=None):
        super(ContinuousFuzzySet, self).__init__()
        self.in_features = in_features
        self._log_widths = None

        # initialize centers
        if centers is None:
            self.centers = torch.nn.parameter.Parameter(torch.randn(self.in_features))
        else:
            centers = self.convert_to_tensor(centers)
            self.centers = torch.nn.parameter.Parameter(centers)

        # initialize widths -- never adjust the widths directly, use the logarithm of them to avoid negatives
        if widths is None:  # we apply the logarithm to the widths, so later, if we train them, and they become
            # nonzero, with an exponential function they are still positive
            # in other words, since gradient descent may make the widths negative, we nullify that effect
            self.widths = torch.rand(self.in_features)
        else:
            # we assume the widths are given to us are within (0, 1)
            widths = self.convert_to_tensor(widths)
            self.widths = torch.nn.parameter.Parameter(torch.abs(widths))

        self.log_widths()  # update the stored log widths

        # initialize support
        if supports is None:
            self.supports = torch.ones(self.in_features)
        else:
            supports = self.convert_to_tensor(supports)
            self.supports = torch.abs(supports)

        # used for feature selection/reduction
        self.special_idx = None

        self.labels = labels

    def log_widths(self):
        with torch.no_grad():
            self._log_widths = torch.nn.parameter.Parameter(torch.log(self.widths))
        return self._log_widths

    @staticmethod
    def convert_to_tensor(values):
        if isinstance(values, torch.Tensor):
            return values
        else:
            return torch.tensor(np.array(values)).float()

    def reshape_parameters(self):
        if self.centers.nelement() == 1:
            self.centers = torch.nn.Parameter(self.centers.reshape(1))
        if self.widths.nelement() == 1:
            self.widths = torch.nn.Parameter(self.widths.reshape(1))
        if self.supports.nelement() == 1:
            self.supports = self.supports.reshape(1)
        self.log_widths()  # update the stored log widths

    def extend(self, centers, widths, supports=None):
        with torch.no_grad():
            self.in_features += len(centers)
            self.reshape_parameters()
            if not isinstance(centers, torch.Tensor):
                centers = torch.tensor(np.array(centers))
            self.centers = torch.nn.Parameter(torch.cat([self.centers, centers]))
            if not isinstance(widths, torch.Tensor):
                widths = torch.tensor(widths)
            self.widths = torch.nn.Parameter(torch.cat([self.widths, widths]))
            if supports is None:
                self.supports = torch.cat([self.supports, torch.ones(len(centers))])
            else:
                if not isinstance(supports, torch.Tensor):
                    supports = torch.tensor(supports)
                self.supports = torch.cat([self.supports, supports])
        self.log_widths()  # update the stored log widths

    def forward(self):
        raise NotImplementedError('The Base Fuzzy Set has no membership function defined.')


class Gaussian(ContinuousFuzzySet):
    """
    Implementation of the Gaussian membership function, written in PyTorch.
    """

    def __init__(self, in_features, centers=None, widths=None, supports=None, labels=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - sigmas: trainable parameter
            centers and sigmas are initialized randomly by default,
            but sigmas must be > 0
        """
        super(Gaussian, self).__init__(in_features, centers, widths, supports, labels)

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
            observations: Two-dimensional matrix of observations, where a row is a single observation and each column
            is related to an attribute measured during that observation.

        Returns:
            The membership degrees of the observations for the Gaussian fuzzy set.
        """
        # https://stackoverflow.com/questions/65022269/how-to-use-a-learnable-parameter-in-pytorch-constrained-between-0-and-1

        return torch.exp(-1.0 * (torch.pow(self.convert_to_tensor(observations).unsqueeze(dim=-1)
                                           - self.centers, 2) / torch.pow(torch.exp(self._log_widths), 2)))


class Triangular(ContinuousFuzzySet):
    """
    Implementation of the Triangular membership function, written in PyTorch.
    """

    def __init__(self, in_features, centers=None, widths=None, supports=None, labels=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - widths: trainable parameter
            centers and widths are initialized randomly by default,
            but widths must be > 0
        """
        super(Triangular, self).__init__(in_features, centers, widths, supports, labels)

    def forward(self, observations):
        """
        Forward pass of the function. Applies the function to the input elementwise.

        Args:
            observations: Two-dimensional matrix of observations, where a row is a single observation and each column
            is related to an attribute measured during that observation.

        Returns:
            The membership degrees of the observations for the Triangular fuzzy set.
        """
        # https://stackoverflow.com/questions/65022269/how-to-use-a-learnable-parameter-in-pytorch-constrained-between-0-and-1
        return torch.max(1.0 - (1.0 / torch.exp(self._log_widths)) * torch.abs(
            self.convert_to_tensor(observations).unsqueeze(dim=-1) - self.centers), torch.tensor(0.0))
