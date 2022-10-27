import torch
import torch.nn as nn

from torch.nn.parameter import Parameter


class AlgebraicProduct(nn.Module):
    """
    Implementation of the Algebraic Product t-norm (Fuzzy AND).
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - importance: trainable parameter
    Examples:
        # >>> a1 = AlgebraicProductTNorm(4)  # 4 inputs from linguistic terms
        # >>> x = torch.FloatTensor(0, 4).uniform(0, 1) #
        # >>> x = a1(x)
    """

    def __init__(self, in_features, importance=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - sigmas: trainable parameter
            importance is initialized to a one vector by default
        """
        super(AlgebraicProduct, self).__init__()
        self.in_features = in_features

        # initialize antecedent importance
        if importance is None:
            self.importance = Parameter(torch.tensor(1.0))
            self.importance.requires_grad = False
        else:
            self.importance = Parameter(torch.abs(torch.tensor(importance)))  # importance can only be [0, 1]
            self.importance.requires_grad = True

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        self.importance = Parameter(torch.abs(self.importance))  # importance can only be [0, 1]
        return torch.prod(torch.mul(torch.tensor(x), self.importance))
