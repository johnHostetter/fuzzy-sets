import torch


class Base(torch.nn.Module):
    def __init__(self, in_features, centers=None, widths=None, supports=None, trainable=True):
        super(Base, self).__init__()
        self.in_features = in_features

        # initialize centers
        if centers is None:
            self.centers = torch.nn.parameter.Parameter(torch.randn(self.in_features))
        else:
            self.centers = torch.tensor(centers)

        # initialize sigmas
        if widths is None:
            with torch.no_grad():
                self.widths = torch.nn.parameter.Parameter(torch.abs(torch.randn(self.in_features)))
        else:
            # we assume the widths are given to us are within (0, 1)
            with torch.no_grad():
                self.widths = torch.abs(torch.tensor(widths))

        # initialize support
        if supports is None:
            self.supports = torch.ones(self.in_features)
        else:
            self.supports = torch.abs(torch.tensor(supports))

        self.centers.requires_grad = trainable
        self.widths.requiresGrad = trainable
        self.centers.grad = None
        self.widths.grad = None
        self.sort()

    def sort(self):
        with torch.no_grad():
            # sorting according to centers
            if self.centers.nelement() > 1:
                sorted_centers, indices = torch.sort(self.centers)
                sorted_widths = self.widths.gather(0, indices.argsort())
                sorted_supports = self.supports.gather(0, indices.argsort())
                self.centers = sorted_centers
                self.widths = sorted_widths
                self.supports = sorted_supports

    def reshape_parameters(self):
        if self.centers.nelement() == 1:
            self.centers = self.centers.reshape(1)
        if self.widths.nelement() == 1:
            self.widths = self.widths.reshape(1)
        if self.supports.nelement() == 1:
            self.supports = self.supports.reshape(1)

    def extend(self, centers, sigmas, supports=None):
        with torch.no_grad():
            self.in_features += len(centers)
            self.reshape_parameters()
            self.centers = torch.cat([self.centers, torch.tensor(centers).reshape(1)])
            self.widths = torch.cat([self.widths, torch.tensor(sigmas).reshape(1)])
            if supports is None:
                self.supports = torch.cat([self.supports, torch.tensor(torch.ones(len(centers)))])
            else:
                self.supports = torch.cat([self.supports, torch.tensor(supports).reshape(1)])
        self.sort()

    def increase_support_of(self, index):
        """
        The 'index' refers to the index of a Gaussian fuzzy set on this dimension.

        We want to increase the support or the count of this fuzzy set as the number of
        data points increase that shows the fuzzy set located at 'index' is satisfactory for representing them.

        Args:
            index:

        Returns:

        """
        values = torch.zeros(self.supports.shape)
        values[index] = 1
        self.supports = torch.add(self.supports, values)

    def forward(self):
        raise NotImplementedError('The Base Fuzzy Set has no membership function defined.')


class Gaussian(Base):
    """
    Implementation of the Gaussian membership function.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - centers: trainable parameter
        - sigmas: trainable parameter
    Examples:
        # >>> a1 = gaussian(256)
        # >>> x = torch.randn(256)
        # >>> x = a1(x)
    """

    def __init__(self, in_features, centers=None, sigmas=None, supports=None, trainable=True):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - sigmas: trainable parameter
            centers and sigmas are initialized randomly by default,
            but sigmas must be > 0
        """
        super(Gaussian, self).__init__(in_features, centers, sigmas, supports, trainable)

    @property
    def sigmas(self):
        return self.widths

    @sigmas.setter
    def sigmas(self, sigmas):
        self.widths = sigmas

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        # https://stackoverflow.com/questions/65022269/how-to-use-a-learnable-parameter-in-pytorch-constrained-between-0-and-1

        return torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(self.sigmas, 2)))


class Triangular(Base):
    """
    Implementation of the Triangular membership function.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - centers: trainable parameter
        - widths: trainable parameter
    Examples:
        # >>> a1 = triangular(256)
        # >>> x = torch.randn(256)
        # >>> x = a1(x)
    """

    def __init__(self, in_features, centers=None, widths=None, supports=None, trainable=True):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - widths: trainable parameter
            centers and widths are initialized randomly by default,
            but widths must be > 0
        """
        super(Triangular, self).__init__(in_features, centers, widths, supports, trainable)

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        # https://stackoverflow.com/questions/65022269/how-to-use-a-learnable-parameter-in-pytorch-constrained-between-0-and-1

        return torch.max(1.0 - (1.0 / self.widths) * torch.abs(x - self.centers), torch.tensor(0.0))
