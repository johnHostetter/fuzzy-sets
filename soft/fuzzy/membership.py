import numpy as np


def gaussian(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))


def triangular(x, center, width):
    """
    Triangular membership function that receives an 'x' value, a
    nd uses the 'params' to determine a degree of membership for 'x'.

    https://www.mathworks.com/help/fuzzy/trimf.html
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    Returns
    -------
    mu : TYPE
        DESCRIPTION.
    """
    values = 1.0 - (1.0 / width) * np.abs(x - center)
    values[(values < 0)] = 0
    return values

    params = {'a': center - width, 'b': center, 'c': center + width}

    if x <= params['a'] or params['c'] <= x:
        mu = 0.0
    elif params['a'] <= x and x <= params['b']:
        mu = (x - params['a']) / (params['b'] - params['a'])
    elif params['b'] <= x <= params['c']:
        mu = (params['c'] - x) / (params['c'] - params['b'])
    else:  # this is a catch-all, and should never be reached, but it is included nonetheless for exception handling
        return None

    return mu
