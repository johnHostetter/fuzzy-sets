"""
Implements the t-norm fuzzy relations.
"""

import torch

from soft.fuzzy.sets.discrete import DiscreteFuzzySet


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


class StandardIntersection(DiscreteFuzzySet):
    """
    A standard intersection of one or more ordinary fuzzy sets.
    """

    def __init__(self, fuzzy_sets, name=None):
        """
        Parameters
        ----------
        fuzzy_sets : 'list'
            A list of elements each of type OrdinaryFuzzySet.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy fets in the same space.
        """
        DiscreteFuzzySet.__init__(self)
        self.fuzzy_sets = fuzzy_sets
        self.name = name

    def degree(self, element):
        """
        Calculates the degree of membership for the provided element value
        where element is a(n) int/float.

        Args:
            element: The element is from the universe of discourse X.

        Returns:
            The degree of membership for the element.
        """
        degrees = []
        for fuzzyset in self.fuzzy_sets:
            degrees.append(fuzzyset.degree(element))
        return min(degrees)


class Minimum:
    # pylint: disable=too-few-public-methods
    """
    A placeholder class for operations expecting the minimum t-norm.
    """
    def __init__(self):
        pass
