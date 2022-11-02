import numpy as np


def gaussian(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))


def triangular(x, center, width):
    """
    Triangular membership function that receives an 'x' value, a
    nd uses the 'center' and 'width' to determine a degree of membership for 'x'.

    https://www.mathworks.com/help/fuzzy/trimf.html
    Args:
        x:
        center:
        width:

    Returns:

    """
    values = 1.0 - (1.0 / width) * np.abs(x - center)
    values[(values < 0)] = 0
    return values
