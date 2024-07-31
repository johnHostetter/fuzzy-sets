"""
Provides functions that help guarantee reproducibility.
"""

import os
import random
import pathlib

import torch
import numpy as np


def set_rng(seed: int) -> None:
    """
    Set the random number generator.

    Args:
        seed: The seed to use for the random number generator.

    Returns:
        None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def path_to_project_root() -> pathlib.Path:
    """
    Return the path to the root of the project.

    Returns:
        The path to the root of the project.
    """
    return pathlib.Path(__file__).parent.parent.parent
