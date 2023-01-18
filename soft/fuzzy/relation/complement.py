from soft.fuzzy.sets.discrete import DiscreteFuzzySet


def StandardComplement(fuzzySet):
    """
    Obtains the standard complement of a fuzzy set as defined by Lotfi A. Zadeh.

    Returns True if successful, else returns False.

    Parameters
    ----------
    fuzzySet : 'OrdinaryDiscreteFuzzySet'

    Returns
    -------
    success : 'bool'
    """

    if isinstance(fuzzySet, DiscreteFuzzySet):
        formulas = []
        for formula in fuzzySet.formulas:
            formula = list(formula)
            formula[0] = 1 - formula[0]
            formula = tuple(formula)
            formulas.append(formula)
        fuzzySet.formulas = formulas
        return True
    return False