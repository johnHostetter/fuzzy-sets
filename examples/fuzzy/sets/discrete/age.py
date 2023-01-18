from soft.fuzzy.relation.snorm import StandardUnion
from soft.fuzzy.relation.tnorm import StandardIntersection
from soft.fuzzy.relation.complement import StandardComplement
from soft.fuzzy.sets.discrete import OrdinaryDiscreteFuzzySet, FuzzyVariable

from sympy import Symbol, Interval, oo  # oo is infinity


def A1():
    """
    A sample construction of a Fuzzy Set called A1.
    """
    formulas = []
    x = Symbol('x')
    formulas.append((1, Interval.Lopen(-oo, 20)))
    formulas.append(((35 - x) / 15, Interval.open(20, 35)))
    formulas.append((0, Interval.Ropen(35, oo)))
    return OrdinaryDiscreteFuzzySet(formulas, 'A1')


def A2():
    """
    A sample construction of a Fuzzy Set called A2.
    """
    formulas = []
    x = Symbol('x')
    formulas.append((0, Interval.Lopen(-oo, 20)))
    formulas.append(((x - 20) / 15, Interval.open(20, 35)))
    formulas.append((1, Interval(35, 45)))
    formulas.append(((60 - x) / 15, Interval.open(45, 60)))
    formulas.append((0, Interval.Ropen(60, oo)))
    return OrdinaryDiscreteFuzzySet(formulas, 'A2')


def A3():
    """
    A sample construction of a Fuzzy Set called A3.
    """
    formulas = []
    x = Symbol('x')
    formulas.append((0, Interval.Lopen(-oo, 45)))
    formulas.append(((x - 45) / 15, Interval.open(45, 60)))
    formulas.append((1, Interval.Ropen(60, oo)))
    return OrdinaryDiscreteFuzzySet(formulas, 'A3')


a1 = A1()
a2 = A2()
a3 = A3()

FuzzyVariable([a1, a2, a3], 'Age').graph(0, 80)
b = StandardIntersection([a1, a2], 'B')
b.graph(0, 80)
c = StandardIntersection([a2, a3], 'C')
c.graph(0, 80)
StandardUnion([b, c], 'B Union C').graph(0, 80)
StandardComplement(a1)
a1.graph(0, 80)
StandardComplement(a3)
a3.graph(0, 80)
StandardIntersection([a1, a3], 'Not(A1) Intersection Not(A3)').graph(0, 80)
StandardComplement(a1)
StandardComplement(a3)
# doesn't work yet
# StandardComplement(StandardUnion([b, c], 'Not (B Union C)')).graph(0, 80)
