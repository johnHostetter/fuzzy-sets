# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:03:47 2020

@author: jhost
"""

from fuzzyset import OrdinaryFuzzySet, FuzzyVariable
from fuzzysetoperations import *
from sympy.solvers import solve
from sympy import Symbol, Interval, oo # oo is infinity
from pynverse import inversefunc
from sympy import lambdify

# https://www-sciencedirect-com.prox.lib.ncsu.edu/science/article/pii/S0957417412008056

def Unknown():
    formulas = []
    x = Symbol('x')
    formulas.append((1, Interval.Lopen(-oo,55)))
    formulas.append((1 - (x-55)/5, Interval.open(55,60)))
    formulas.append((0, Interval.Ropen(60,oo)))
    return OrdinaryFuzzySet(formulas, 'Unknown')

def Known():
    formulas = []
    x = Symbol('x')
    formulas.append(((x-70)/5, Interval.open(70,75)))
    formulas.append((1, Interval(75,85)))
    formulas.append((1-(x-85)/5, Interval.open(85,90)))
    formulas.append((0, Interval.Lopen(-oo,70)))
    formulas.append((0, Interval.Ropen(90,oo)))
    return OrdinaryFuzzySet(formulas, 'Known')

def UnsatisfactoryUnknown():
    formulas = []
    x = Symbol('x')
    formulas.append(((x-55)/5, Interval.open(55,60)))
    formulas.append((1, Interval(60,70)))
    formulas.append((1-(x-70)/5, Interval.open(70,75)))
    formulas.append((0, Interval.Lopen(-oo,55)))
    formulas.append((0, Interval.Ropen(75, oo)))
    return OrdinaryFuzzySet(formulas, 'Unsatisfactory Unknown')

def Learned():
    formulas = []
    x = Symbol('x')
    formulas.append(((x-85)/5, Interval.open(85,90)))
    formulas.append((1, Interval(90,100)))
    formulas.append((0, Interval.Lopen(-oo,85)))
    return OrdinaryFuzzySet(formulas, 'Learned')

unknown = Unknown()
known = Known()
unsatisfactoryUnknown = UnsatisfactoryUnknown()
learned = Learned()

fuzzyVariable = FuzzyVariable([unknown, known, unsatisfactoryUnknown, learned], 'Student Knowledge')
#fuzzyVariable.graph(samples=50)
z = fuzzyVariable.degree(87)
#StandardIntersection(fuzzyVariable.fuzzySets[1:2]).graph()

print(isinstance(unknown.formulas[1][0], sympy.Expr))

y = inversefunc(lambdify(Symbol('x'), unknown.formulas[1][0], 'numpy'), y_values=0.001)

alphacut = AlphaCut(known, 0.5, 'AlphaCut')
alphacut.graph(60,100)