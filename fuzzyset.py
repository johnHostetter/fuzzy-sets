# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:40:22 2020

@author: jhost
"""

import numpy as np
from sympy import Symbol
import matplotlib.pyplot as plt

# https://docs.sympy.org/latest/modules/integrals/integrals.html
# https://docs.sympy.org/latest/modules/sets.html

class FuzzySet:
    def __init__(self):
        pass
    def graph(self, lower=0, upper=100, samples=100):
        """
        Graphs the Fuzzy Set in the universe of elements.
        
        Accepts an optional lower and upper bound. If left unspecified, the lower bound will be
        zero and the upper bound will be one hundred. The samples parameter specifies the number
        of x values to test in the domain to approximate the graph. A higher sample value will
        yield a higher resolution of the graph, but large values will lead to performance issues.
        Default value for samples is one hundred.
        """
        x_list = np.linspace(lower, upper, samples)
        y_list = []
        for x in x_list:
            y_list.append(self.degree(x))
        if self.name != None:    
            plt.title('%s Fuzzy Set' % self.name)
        else:
            plt.title('Unnamed Fuzzy Set')
        
        plt.axes()
        plt.xlim([lower, upper])
        plt.ylim([0, 1.1])
        plt.xlabel('Elements of Universe')
        plt.ylabel('Degree of Membership')
        plt.plot(x_list, y_list, color='grey', label='mu')
        plt.legend()
        plt.show()
    
class OrdinaryFuzzySet(FuzzySet):
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
                    A = OrdinaryFuzzySet(formulas)
        """
        FuzzySet.__init__(self)
        self.formulas = formulas
        self.name = name
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
        formula = self.fetch(x)[0]
        try:
            y = float(formula.subs(Symbol('x'), x))
        except AttributeError:
            y = formula
        return y
    
class FuzzyVariable(FuzzySet):
    def __init__(self, fuzzySets, name=None):
        FuzzySet.__init__(self)
        self.fuzzySets = fuzzySets
        self.name = name
    def graph(self, lower=0, upper=100, samples=100):
        """
        Graphs the Fuzzy Set in the universe of elements.
        
        Accepts an optional lower and upper bound. If left unspecified, the lower bound will be
        zero and the upper bound will be one hundred. The samples parameter specifies the number
        of x values to test in the domain to approximate the graph. A higher sample value will
        yield a higher resolution of the graph, but large values will lead to performance issues.
        Default value for samples is one hundred.
        """
        for fuzzySet in self.fuzzySets:
            x_list = np.linspace(lower, upper, samples)
            y_list = []
            for x in x_list:
                y_list.append(fuzzySet.degree(x))
            plt.plot(x_list, y_list, color=np.random.rand(3,), label=fuzzySet.name)

        if self.name != None:    
            plt.title('%s Fuzzy Variable' % self.name)
        else:
            plt.title('Unnamed Fuzzy Variable')
        
        plt.axes()
        plt.xlim([lower, upper])
        plt.ylim([0, 1.1])
        plt.xlabel('Elements of Universe')
        plt.ylabel('Degree of Membership')
        plt.legend()
        plt.show()