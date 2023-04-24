import sympy

from soft.fuzzy.sets.discrete import DiscreteFuzzySet


class SpecialFuzzySet(DiscreteFuzzySet):
    """
    The special fuzzy set membership function for a given element x in the universe of
    discourse X, is defined as the alpha value multipled by the element x's degree of
    membership within the fuzzy set's alpha cut.
    """

    def __init__(self, fuzzyset, alpha, name=None):
        """
        Parameters
        ----------
            fuzzyset : 'OrdinaryDiscreteFuzzySet'
                An ordinary fuzzy set to retrieve the special fuzzy set given the alpha.
            alpha : 'float'
                The alpha value that elements' membership degree must exceed or be equal to.
            name : 'str'/'None'
                Default value is None. Allows the user to specify the name of the fuzzy set.
                This feature is useful when visualizing the fuzzy set, and its interaction with
                other fuzzy sets in the same space.
        """
        alphaCut = AlphaCut(fuzzyset, alpha)
        sympy.Interval = alphaCut.formulas[0][1]
        for formula in alphaCut.formulas[1:]:
            sympy.Interval = sympy.Union(sympy.Interval, formula[1])
        self.formulas = [(alpha, sympy.Interval)]
        self.alpha = alpha
        self.name = name

    def fetch(self, x):
        """
        Fetch the corresponding formula for the provided x value where x is a(n) int/float.

        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.

        Returns
        -------
        formula : 'tuple'/'None'
            Returns the tuple containing the formula and corresponding sympy.Interval. Returns
            None if a formula for the element x could not be found.
        """
        for formula in self.formulas:
            if formula[1].contains(
                x
            ):  # check the formula's sympy.Interval to see if it contains x
                return formula
        return None

    def degree(self, x):
        """
        Calculates the degree of membership for the provided x value where x is a(n) int/float.

        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.

        Returns
        -------
        y : 'float'
            The degree of membership for element x.
        """
        result = self.fetch(x)
        if result is not None:
            formula = result[0]
        else:
            return 0
        try:
            y = float(formula.subs(sympy.Symbol("x"), x))
        except AttributeError:
            y = formula
        return y

    def height(self):
        """
        Calculates the height of the special fuzzy set.

        Returns
        -------
        height : 'float'
            The height, or supremum, of the Special Fuzzy Set.
        """
        return self.alpha


class AlphaCut(DiscreteFuzzySet):
    """
    The alpha cut of a fuzzy set yields a crisp set.
    """

    def __init__(self, fuzzyset, alpha, name=None):
        """
        Parameters
        ----------
        formulas : 'list'
            A list of 2-tuples. The first element in the tuple at index 0 is the formula
            equal to f(x) and the second element in the tuple at index 1 is the sympy.Interval
            where the formula in the tuple is valid.
        alpha : 'float'
            The alpha value that elements' membership degree must exceed or be equal to.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy sets in the same space.
        """
        self.alpha = alpha
        self.name = name
        formulas = []
        for formula in fuzzyset.formulas:
            if isinstance(formula[0], sympy.Expr):
                # x = inversefunc(
                #     lambdify(sympy.Symbol("x"), formula[0], "numpy"), y_values=alpha
                # )
                interval_1 = sympy.solve(alpha <= formula[0])
                interval_2 = sympy.solve(formula[0] <= alpha)
                x_interval = sympy.Intersection(interval_1.as_set(), interval_2.as_set())
                x = x_interval.atoms().pop()
                if formula[1].contains(x):
                    # the x is within the sympy.Interval, now check the direction
                    y = formula[0].subs(sympy.Symbol("x"), x - (1e-6))
                    if y >= alpha:
                        # then all values less than or equal to x are valid
                        if formula[1].left_open:
                            sympy.Interval = sympy.Interval.Lopen(formula[1].inf, x)
                        else:
                            sympy.Interval = sympy.Interval(formula[1].inf, x)
                    else:
                        # then all values greater than or equal to x are valid
                        if formula[1].right_open:
                            sympy.Interval = sympy.Interval.Ropen(x, formula[1].sup)
                        else:
                            sympy.Interval = sympy.Interval(x, formula[1].sup)
                    formula = list(formula)
                    formula[1] = sympy.Interval
                    formula = tuple(formula)
                    formulas.append(formula)
            else:
                if formula[0] >= alpha:
                    formulas.append(formula)
        self.formulas = formulas

    def fetch(self, x):
        """
        Fetch the corresponding formula for the provided x value where x is a(n) int/float.

        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.

        Returns
        -------
        formula : 'tuple'/'None'
            Returns the tuple containing the formula and corresponding sympy.Interval. Returns
            None if a formula for the element x could not be found.
        """
        for formula in self.formulas:
            if formula[1].contains(
                x
            ):  # check the formula's sympy.Interval to see if it contains x
                return formula
        return None

    def degree(self, x):
        """
        Calculates the degree of membership for the provided x value where x is a(n) int/float.

        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.

        Returns
        -------
        y : 'float'
            The degree of membership for element x.
        """
        result = self.fetch(x)
        if result is not None:
            formula = result[0]
        else:
            return 0
        try:
            y = float(formula.subs(sympy.Symbol("x"), x))
        except AttributeError:
            y = formula
        return y
