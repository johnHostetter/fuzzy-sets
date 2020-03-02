# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:49:59 2020

@author: jhost
"""
import sympy
from fuzzyset import FuzzySet
from pynverse import inversefunc
from sympy import lambdify, Symbol, Interval

class AlphaCut(FuzzySet):
    def __init__(self, fuzzyset, alpha, name=None):
        self.fuzzyset = fuzzyset
        self.name = name
        formulas = []
        for formula in fuzzyset.formulas:
            if isinstance(formula[0], sympy.Expr):
                x = inversefunc(lambdify(Symbol('x'), formula[0], 'numpy'), y_values=alpha)
                if formula[1].contains(x):
                    # the x is within the interval, now check the direction
                    y = formula[0].subs(Symbol('x'), x - (1e-6))
                    if y >= alpha:
                        # then all values less than or equal to x are valid
                        if formula[1].left_open:    
                            interval = Interval.Lopen(formula[1].inf, x)
                        else:
                            interval = Interval(formula[1].inf, x)
                    else:
                        # then all values greater than or equal to x are valid
                        if formula[1].right_open:
                            interval = Interval.Ropen(x, formula[1].sup)
                        else:
                            interval = Interval(x, formula[1].sup)
                    formula = list(formula)
                    formula[1] = interval
                    formula = tuple(formula)
                    formulas.append(formula)
            else:
                if formula[0] >= alpha:
                    formulas.append(formula)
        self.formulas = formulas
    def fetch(self, x):
        """ 
        Fetch the corresponding formula for the provided x value where x is a(n) int/float.
        
        Returns None if a formula could not be found.
        """
        for formula in self.formulas:
            if formula[1].contains(x): # check the formula's interval to see if it contains x
                return formula
        return None
    def degree(self, x):
        result = self.fetch(x)
        if result != None:
            formula = result[0]
        else:
            return 0
        try:
            y = float(formula.subs(Symbol('x'), x))
        except AttributeError:
            y = formula
        return y
        
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