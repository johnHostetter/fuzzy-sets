import torch


class Base(torch.nn.Module):
    def __init__(self, in_features, centers=None, widths=None, supports=None, labels=None,
                 trainable=True, sort_by='centers'):
        super(Base, self).__init__()
        self.in_features = in_features
        self._log_widths = None
        self.sort_by = sort_by

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
            with torch.no_grad():
                self.widths = torch.rand(self.in_features)
        else:
            # we assume the widths are given to us are within (0, 1)
            with torch.no_grad():
                widths = self.convert_to_tensor(widths)
                self.widths = torch.abs(widths)

        self.log_widths()  # update the stored log widths

        # initialize support
        if supports is None:
            self.supports = torch.ones(self.in_features)
        else:
            supports = self.convert_to_tensor(supports)
            self.supports = torch.abs(supports)

        self.labels = labels
        self.trainable = self.train(mode=trainable)
        if sort_by == 'centers':
            self.sort()

    def log_widths(self):
        with torch.no_grad():
            self._log_widths = torch.nn.parameter.Parameter(torch.log(self.widths))
        return self._log_widths

    @staticmethod
    def convert_to_tensor(values):
        if isinstance(values, torch.Tensor):
            return values
        else:
            return torch.tensor(values).float()

    def train(self, mode):
        """
        Disable/enable training for the granules' parameters.
        """
        self.centers.requires_grad = mode
        self._log_widths.requiresGrad = mode
        if not mode:
            self.centers.grad = None
            self._log_widths.grad = None
        return mode

    def sort(self):
        pass
        # with torch.no_grad():
        #     # sorting according to centers
        #     if self.centers.nelement() > 1:
        #         sorted_centers, indices = torch.sort(self.centers)
        #         sorted_widths = self.widths.gather(0, indices.argsort())
        #         sorted_supports = self.supports.gather(0, indices.argsort())
        #         self.centers = torch.nn.Parameter(sorted_centers)
        #         self.widths = sorted_widths
        #         self.supports = sorted_supports
        # self.log_widths()  # update the stored log widths
        # self.train(self.trainable)

    def reshape_parameters(self):
        if self.centers.nelement() == 1:
            self.centers = torch.nn.Parameter(self.centers.reshape(1))
        if self.widths.nelement() == 1:
            self.widths = self.widths.reshape(1)
        if self.supports.nelement() == 1:
            self.supports = self.supports.reshape(1)
        self.log_widths()  # update the stored log widths
        self.train(self.trainable)

    def extend(self, centers, widths, supports=None):
        with torch.no_grad():
            self.in_features += len(centers)
            self.reshape_parameters()
            try:
                self.centers = torch.nn.Parameter(torch.cat([self.centers, torch.tensor(centers).reshape(1)]))
            except RuntimeError:  # RuntimeError: shape '[1]' is invalid for input of size 4
                self.centers = torch.nn.Parameter(torch.cat([self.centers, torch.tensor(centers)]))

            try:
                self.widths = torch.cat([self.widths, torch.tensor(widths).reshape(1)])
            except ValueError:
                print((self.widths, widths))
                self.widths = torch.cat([self.widths, torch.tensor(widths)])
            if supports is None:
                self.supports = torch.cat([self.supports, torch.ones(len(centers))])
            else:
                try:
                    self.supports = torch.cat([self.supports, torch.tensor(supports).reshape(1)])
                except RuntimeError:
                    self.supports = torch.cat([self.supports, torch.tensor(supports)])
        self.log_widths()  # update the stored log widths
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
        with torch.no_grad():
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

    def __init__(self, in_features, centers=None, widths=None, supports=None, labels=None,
                 trainable=True, sort_by='centers'):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - sigmas: trainable parameter
            centers and sigmas are initialized randomly by default,
            but sigmas must be > 0
        """
        super(Gaussian, self).__init__(in_features, centers, widths, supports, labels, trainable, sort_by)

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

        log_results = torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(torch.exp(self._log_widths), 2)))
        no_log_results = torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(self.widths, 2)))
        # if not torch.allclose(log_results, no_log_results, rtol=1e-01, atol=1e-01):
        #     print('with logs: {}'.format(log_results))
        #     print('w/o logs: {}'.format(no_log_results))
        #     print('stop')
        #     quit()
        return no_log_results
        # return torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(torch.exp(self._log_widths), 2)))


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

    def __init__(self, in_features, centers=None, widths=None, supports=None, labels=None,
                 trainable=True, sort_by='centers'):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - widths: trainable parameter
            centers and widths are initialized randomly by default,
            but widths must be > 0
        """
        super(Triangular, self).__init__(in_features, centers, widths, supports, labels, trainable, sort_by)

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        # https://stackoverflow.com/questions/65022269/how-to-use-a-learnable-parameter-in-pytorch-constrained-between-0-and-1

        return torch.max(1.0 - (1.0 / torch.exp(self._log_widths)) * torch.abs(x - self.centers), torch.tensor(0.0))
