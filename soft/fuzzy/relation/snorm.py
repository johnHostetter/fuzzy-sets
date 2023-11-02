"""
Implements the s-norm fuzzy relations.
"""

from typing import Union, List, Callable

from soft.fuzzy.sets.discrete import DiscreteFuzzySet
from soft.fuzzy.relation.extension import DiscreteFuzzyRelation


class StandardUnion(DiscreteFuzzyRelation):
    """
    A standard union of one or more ordinary fuzzy sets.
    """

    def __init__(self, fuzzy_sets: List[DiscreteFuzzySet], name=None):
        """
        Parameters
        ----------
        formulas : 'list'
            A list of elements each of type OrdinaryDiscreteFuzzySet.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy fets in the same space.
        """
        DiscreteFuzzyRelation.__init__(self, formulas=fuzzy_sets, name=name)
        self.fuzzy_sets = fuzzy_sets

    def degree(self, element: Union[int, float], mode: Callable = max):
        """
        Calculates the degree of membership for the provided element value
        where element is a(n) int/float.

        Args:
            element: The element is from the universe of discourse X.
            mode: The mode of the degree of membership; the default is max.

        Returns:
            The degree of membership for the element.
        """
        return self.degree(element, mode)
