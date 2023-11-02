"""
Implements the t-norm fuzzy relations.
"""
from typing import List

import torch

from soft.fuzzy.sets.discrete import DiscreteFuzzySet
from soft.fuzzy.relation.extension import DiscreteFuzzyRelation


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


class StandardIntersection(DiscreteFuzzyRelation):
    """
    A standard intersection of one or more ordinary fuzzy sets.
    """

    def __init__(self, fuzzy_sets: List[DiscreteFuzzySet], name=None):
        """
        Parameters
        ----------
        fuzzy_sets : 'list'
            A list of elements each of type OrdinaryDiscreteFuzzySet.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy fets in the same space.
        """
        DiscreteFuzzyRelation.__init__(self, formulas=fuzzy_sets, name=name, mode=min)
        self.fuzzy_sets = fuzzy_sets


class Minimum:
    # pylint: disable=too-few-public-methods
    """
    A placeholder class for operations expecting the minimum t-norm.
    """

    def __init__(self):
        pass
