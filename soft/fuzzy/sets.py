import torch


class Gaussian(torch.nn.Module):
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

    def __init__(self, in_features, centers=None, sigmas=None, trainable=True):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - sigmas: trainable parameter
            centers and sigmas are initialized randomly by default,
            but sigmas must be > 0
        """
        super(Gaussian, self).__init__()
        self.in_features = in_features

        # initialize centers
        if centers is None:
            self.centers = torch.nn.parameter.Parameter(torch.randn(self.in_features))
        else:
            self.centers = torch.tensor(centers)

        # initialize sigmas
        if sigmas is None:
            self.sigmas = torch.nn.parameter.Parameter(torch.abs(torch.randn(self.in_features)))
        else:
            # we assume the sigmas are given to us are within (0, 1)
            self.sigmas = torch.abs(torch.tensor(sigmas))

        self.centers.requires_grad = trainable
        self.sigmas.requiresGrad = trainable
        self.centers.grad = None
        self.sigmas.grad = None

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        # https://stackoverflow.com/questions/65022269/how-to-use-a-learnable-parameter-in-pytorch-constrained-between-0-and-1

        return torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(self.sigmas, 2)))
