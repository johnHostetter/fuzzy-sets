# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:40:22 2020

@author: jhost
"""

import numpy as np
from sympy.solvers import solve
from sympy import Symbol, Interval, oo # oo is infinity
import matplotlib.pyplot as plt

# https://docs.sympy.org/latest/modules/integrals/integrals.html
# https://docs.sympy.org/latest/modules/sets.html

class FuzzySet:
    def __init__(self, formulas, name=None):
        """ 
        The constructor expects a list of 2-tuples, where the first element in the
        tuple at index 0 is the formula equal to f(x) and the second element in the tuple
        at index 1 is the Interval such that the formula in the tuple holds true for.
        
        Formulas should be organized in the list such that the formulas and their corresponding
        intervals are specified from smallest x values to largest x values.
        
        The formulas provided are the piece-wise function of the membership function specifying
        the Fuzzy Set.
        
        The optional parameter called name allows the user to specify the name of the Fuzzy Set.
        This feature is useful for when visualizing the Fuzzy Set and its interaction with other
        Fuzzy Sets in the same space.
        
        Example:
                    formulas = []
                    x = Symbol('x')
                    formulas.append((1, Interval.Lopen(-oo,20)))
                    formulas.append(((35-x)/15,Interval.open(20,35)))
                    formulas.append((0, Interval.Ropen(35,oo)))
                    A = FuzzySet(formulas)
        """
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

#x = np.linspace(-2.5, 2.5, 1000)
#x = Symbol('x')

#z = np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])


    
#plt.axes()
#plt.xlim([-2.5, 2.5])
#plt.xlabel('Elements of Universe')
#plt.ylabel('Degree of Membership')
#plt.plot(x, z, color='grey')
#plt.legend()
#plt.show()