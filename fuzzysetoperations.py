# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:49:59 2020

@author: jhost
"""

from fuzzyset import FuzzySet
from sympy import Symbol, Interval, oo # oo is infinity

def StandardComplement(A):
    """
    Obtains the standard complement of a Fuzzy Set A as defined by Lotfi A. Zadeh.
    
    Returns True if successful, else returns False.
    
    Example:
                StandardComplement(A)
    """
    if isinstance(A, FuzzySet):
        formulas = []
        for formula in A.formulas:
            formula = list(formula)
            formula[0] = 1- formula[0]
            formula = tuple(formula)
            formulas.append(formula)
        A.formulas = formulas
        return True
    return False
            

def A1():
    """
    A sample construction of a Fuzzy Set called A1.
    """
    formulas = []
    x = Symbol('x')
    formulas.append((1, Interval.Lopen(-oo,20)))
    formulas.append(((35-x)/15,Interval.open(20,35)))
    formulas.append((0, Interval.Ropen(35,oo)))
    return FuzzySet(formulas, 'A1')

a1 = A1()

a1.graph(0,80)

StandardComplement(a1)

a1.graph(0,80)
