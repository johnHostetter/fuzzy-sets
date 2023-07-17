"""
Provides functions that help guarantee reproducibility.
"""
import os
import random
import pathlib
from typing import Union

import torch
import numpy as np

from YACS.yacs import Config


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


def env_seed(env, seed: int) -> None:
    """
    Set the random number generator, and also set the random number generator for gym.env.

    Args:
        env: The environment.
        seed: The seed to use for the random number generator.

    Returns:
        None
    """
    set_rng(seed)
    try:
        env.reset(seed=seed)
    except TypeError:  # older version of gym (e.g., 0.21)
        env.seed(seed)
    env.action_space.seed(seed)


def load_configuration(
    file_name: Union[str, pathlib.Path] = "default_config.yaml",
    convert_data_types: bool = True
) -> Config:
    """
    Load and return the default configuration that should be used for models, if another
    overriding configuration is not used in its place.

    Args:
        file_name: Union[str, pathlib.Path] Either a file name (str) where the function will look up
        the *.yml configuration file on the parent directory (i.e., git repository) level, or a
        pathlib.Path where the object redirects the function to a specific location that may be in
        a subdirectory of this repository.
        convert_data_types: Whether to convert or map string values to their true values. For
        example, convert "golden" into the golden ratio.

    Returns:
        The configuration settings.
    """
    file_path = pathlib.Path(__file__).parent.parent.parent / file_name
    config = Config(str(file_path))
    if convert_data_types:
        return parse_configuration(config)
    return config


def parse_configuration(config: Config) -> Config:
    """
    Given the configuration, parse through its values and convert them to their true values.
    For example, convert "golden" into the golden ratio.

    Args:
        config: The configuration settings.

    Returns:
        The updated configuration settings.
    """
    with config.unfreeze():
        if config.fuzzy.t_norm.yager.lower() == "euler":
            w_parameter = np.e
        elif config.fuzzy.t_norm.yager.lower() == "golden":
            w_parameter = (1 + 5 ** 0.5) / 2
        else:
            w_parameter = float(config.fuzzy.t_norm.yager)
        config.fuzzy.t_norm.yager = w_parameter
    return config
