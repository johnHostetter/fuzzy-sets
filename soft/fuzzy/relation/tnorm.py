import torch

from soft.fuzzy.sets import DiscreteFuzzySet


class AlgebraicProduct(torch.nn.Module):
    """
    Implementation of the Algebraic Product t-norm (Fuzzy AND).
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
            self.importance = torch.nn.parameter.Parameter(torch.tensor(1.0))
            self.importance.requires_grad = False
        else:
            self.importance = torch.nn.parameter.Parameter(torch.abs(torch.tensor(importance)))  # importance can only be [0, 1]
            self.importance.requires_grad = True

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        self.importance = torch.nn.parameter.Parameter(torch.abs(self.importance))  # importance can only be [0, 1]
        return torch.prod(torch.mul(torch.tensor(x), self.importance))


class StandardIntersection(DiscreteFuzzySet):
    """
    A standard intersection of one or more ordinary fuzzy sets.
    """

    def __init__(self, fuzzysets, name=None):
        """
        Parameters
        ----------
        fuzzySets : 'list'
            A list of elements each of type OrdinaryFuzzySet.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy fets in the same space.
        """
        DiscreteFuzzySet.__init__(self)
        self.fuzzysets = fuzzysets
        self.name = name

    def degree(self, x):
        """
        Calculates the degree of membership for the provided x value where x is a(n) int/float.

        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.

        Returns
        -------
        y : 'float'
            The degree of membership for element x.
        """
        degrees = []
        for fuzzyset in self.fuzzysets:
            degrees.append(fuzzyset.degree(x))
        return min(degrees)
    