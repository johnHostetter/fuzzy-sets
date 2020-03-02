# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:49:59 2020

@author: jhost
"""
from fuzzyset import FuzzySet

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

class StandardUnion(FuzzySet):
    def __init__(self, fuzzysets, name=None):
        FuzzySet.__init__(self)
        self.fuzzysets = fuzzysets
        self.name = name
    def degree(self, x):
        degrees = []
        for fuzzyset in self.fuzzysets:
            degrees.append(fuzzyset.degree(x))
        return max(degrees)
        
class StandardIntersection(FuzzySet):
    def __init__(self, fuzzysets, name=None):
        FuzzySet.__init__(self)
        self.fuzzysets = fuzzysets
        self.name = name
    def degree(self, x):
        degrees = []
        for fuzzyset in self.fuzzysets:
            degrees.append(fuzzyset.degree(x))
        return min(degrees)