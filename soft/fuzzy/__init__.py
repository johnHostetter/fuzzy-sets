from enum import Enum


class TNorm(Enum):
    """
    Enumerates the types of t-norms.
    """

    PRODUCT = "product"  # i.e., algebraic product
    MINIMUM = "minimum"
    ACZEL_ALSINA = "aczel_alsina"  # not yet implemented
    SOFTMAX_SUM = "softmax_sum"
    SOFTMAX_MEAN = "softmax_mean"
    LUKASIEWICZ = "generalized_lukasiewicz"
    # the following are to be implemented
    DRASTIC = "drastic"
    NILPOTENT = "nilpotent"
    HAMACHER = "hamacher"
    EINSTEIN = "einstein"
    YAGER = "yager"
    DUBOIS = "dubois"
    DIF = "dif"
