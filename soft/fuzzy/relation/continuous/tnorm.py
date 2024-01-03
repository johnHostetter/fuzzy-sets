"""
Implements the t-norm fuzzy relations.
"""
import torch


class AlgebraicProduct(torch.nn.Module):
    """
    Implementation of the Algebraic Product t-norm (Fuzzy AND).
    """

    def __init__(self, in_features=None, importance=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - sigmas: trainable parameter
            importance is initialized to a one vector by default
        """
        super().__init__()
        self.in_features = in_features

        # initialize antecedent importance
        if importance is None:
            self.importance = torch.nn.parameter.Parameter(torch.tensor(1.0))
            self.importance.requires_grad = False
        else:
            if not isinstance(importance, torch.Tensor):
                importance = torch.Tensor(importance)
            self.importance = torch.nn.parameter.Parameter(
                torch.abs(importance)
            )  # importance can only be [0, 1]
            self.importance.requires_grad = True

    def forward(self, elements):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        self.importance = torch.nn.parameter.Parameter(
            torch.abs(self.importance)
        )  # importance can only be [0, 1]
        return torch.prod(torch.mul(elements, self.importance))


class Minimum:
    # pylint: disable=too-few-public-methods
    """
    A placeholder class for operations expecting the minimum t-norm.
    """

    def __init__(self):
        pass
