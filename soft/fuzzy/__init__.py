from enum import Enum


class TNorm(Enum):
    """
    Enumerates the types of t-norms.
    """

    PRODUCT = "product"  # i.e., algebraic product
    MINIMUM = "minimum"
