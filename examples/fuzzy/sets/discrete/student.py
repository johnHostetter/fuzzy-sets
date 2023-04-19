"""
Demo of working with discrete fuzzy sets for a toy task regarding knowledge of material.
"""
from sympy import Symbol, Interval, oo  # oo is infinity

from soft.fuzzy.relation.tnorm import StandardIntersection
from soft.fuzzy.relation.extension import AlphaCut, SpecialFuzzySet
from soft.fuzzy.sets.discrete import OrdinaryDiscreteFuzzySet, FuzzyVariable


# https://www-sciencedirect-com.prox.lib.ncsu.edu/science/article/pii/S0957417412008056

def unknown():
    """
    Create a fuzzy set for the linguistic term 'unknown'.

    Returns:
        OrdinaryDiscreteFuzzySet
    """
    formulas = []
    element = Symbol('x')
    formulas.append((1, Interval.Lopen(-oo, 55)))
    formulas.append((1 - (element - 55) / 5, Interval.open(55, 60)))
    formulas.append((0, Interval.Ropen(60, oo)))
    return OrdinaryDiscreteFuzzySet(formulas, 'unknown')


def known():
    """
    Create a fuzzy set for the linguistic term 'known'.

    Returns:
        OrdinaryDiscreteFuzzySet
    """
    formulas = []
    element = Symbol('x')
    formulas.append(((element - 70) / 5, Interval.open(70, 75)))
    formulas.append((1, Interval(75, 85)))
    formulas.append((1 - (element - 85) / 5, Interval.open(85, 90)))
    formulas.append((0, Interval.Lopen(-oo, 70)))
    formulas.append((0, Interval.Ropen(90, oo)))
    return OrdinaryDiscreteFuzzySet(formulas, 'known')


def unsatisfactory_unknown():
    """
    Create a fuzzy set for the linguistic term 'unsatisfactory unknown'.

    Returns:
        OrdinaryDiscreteFuzzySet
    """
    formulas = []
    element = Symbol('x')
    formulas.append(((element - 55) / 5, Interval.open(55, 60)))
    formulas.append((1, Interval(60, 70)))
    formulas.append((1 - (element - 70) / 5, Interval.open(70, 75)))
    formulas.append((0, Interval.Lopen(-oo, 55)))
    formulas.append((0, Interval.Ropen(75, oo)))
    return OrdinaryDiscreteFuzzySet(formulas, 'unsatisfactory unknown')


def learned():
    """
    Create a fuzzy set for the linguistic term 'learned'.

    Returns:
        OrdinaryDiscreteFuzzySet
    """
    formulas = []
    element = Symbol('x')
    formulas.append(((element - 85) / 5, Interval.open(85, 90)))
    formulas.append((1, Interval(90, 100)))
    formulas.append((0, Interval.Lopen(-oo, 85)))
    return OrdinaryDiscreteFuzzySet(formulas, 'learned')


if __name__ == "__main__":
    terms = [unknown(), known(), unsatisfactory_unknown(), learned()]

    fuzzyVariable = FuzzyVariable(terms, 'Student Knowledge')
    fuzzyVariable.graph(samples=150)

    # --- DEMO --- Classify 'element'

    example_element_membership = fuzzyVariable.degree(element=73)

    alphacut = AlphaCut(known(), 0.6, 'AlphaCut')
    alphacut.graph(samples=250)
    specialFuzzySet = SpecialFuzzySet(known(), 0.5, 'Special')
    specialFuzzySet.graph()

    cuts = []
    idx, MAX_HEIGHT, IDX_OF_MAX = 0, 0, 0
    for idx, membership_to_fuzzy_term in enumerate(example_element_membership):
        if example_element_membership[idx] > 0:
            specialFuzzySet = SpecialFuzzySet(
                terms[idx], membership_to_fuzzy_term, terms[idx].name)
            cuts.append(StandardIntersection([specialFuzzySet, terms[idx]], name=f"A{idx + 1}"))

            # maximum membership principle
            if membership_to_fuzzy_term > MAX_HEIGHT:
                MAX_HEIGHT, IDX_OF_MAX = membership_to_fuzzy_term, idx

    Z = FuzzyVariable(cuts, name='Confluence')
    Z.graph()

    # maximum membership principle
    print(f"Maximum Membership Principle: {terms[idx].name}")
