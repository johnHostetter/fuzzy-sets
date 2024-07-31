"""
Test extensions of the BaseDiscreteFuzzySet, such as AlphaCut or SpecialFuzzySet.
"""

import unittest

from fuzzy_relations.discrete.snorm import StandardUnion
from fuzzy_relations.discrete.tnorm import StandardIntersection
from fuzzy_relations.discrete.extension import AlphaCut, SpecialFuzzySet
from examples.discrete.student import known, learned


class TestAlphaCut(unittest.TestCase):
    def test_alpha_cut(self):
        alpha_cut = AlphaCut(known(), 0.6, "AlphaCut")
        assert alpha_cut.degree(80) == alpha_cut.alpha


class TestSpecialFuzzySet(unittest.TestCase):
    def test_special_fuzzy_set(self):
        special_fuzzy_set = SpecialFuzzySet(known(), 0.5, "Special")
        assert special_fuzzy_set.height() == special_fuzzy_set.alpha
        assert special_fuzzy_set.degree(80) == special_fuzzy_set.alpha


class TestDiscreteFuzzyRelation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terms = [known(), learned()]

    def test_standard_intersection(self):
        standard_intersection = StandardIntersection(fuzzy_sets=self.terms)
        assert standard_intersection.degree(87) == 0.4

    def test_standard_union(self):
        standard_union = StandardUnion(fuzzy_sets=self.terms)
        assert standard_union.degree(87) == 0.6
