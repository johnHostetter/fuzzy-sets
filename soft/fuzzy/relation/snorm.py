from soft.fuzzy.sets.discrete import DiscreteFuzzySet


class StandardUnion(DiscreteFuzzySet):
    """
    A standard union of one or more ordinary fuzzy sets.
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
        return max(degrees)
