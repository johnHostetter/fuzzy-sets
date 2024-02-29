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
from soft.fuzzy.relation.continuous.tnorm import AlgebraicProduct, Minimum


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
    env.reset(seed=seed)  # for older version of gym (e.g., 0.21) use env.seed(seed)
    env.action_space.seed(seed)


def path_to_project_root() -> pathlib.Path:
    """
    Return the path to the root of the project.

    Returns:
        The path to the root of the project.
    """
    return pathlib.Path(__file__).parent.parent.parent


def load_configuration(
    file_name: Union[str, pathlib.Path] = "default_configuration.yaml",
    convert_data_types: bool = True,
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
    file_path = path_to_project_root() / "configurations" / file_name
    config = Config(str(file_path))
    torch.set_default_device("cpu")
    if convert_data_types:
        return parse_configuration(config)
    return config


def parse_configuration(config: Config, reverse=False) -> Config:
    """
    Given the configuration, parse through its values and convert them to their true values.
    For example, convert "golden" into the golden ratio.

    Args:
        config: The configuration settings.
        reverse: Reverse the parsing of the configuration values; used for when saving
            configuration settings of a KnowledgeBase.

    Returns:
        The updated configuration settings.
    """
    with config.unfreeze():
        config.training.learning_rate = float(config.training.learning_rate)
        if reverse:
            values_to_str = {
                # np.e: "euler",
                # (1 + 5**0.5) / 2: "golden",
                AlgebraicProduct: "algebraic_product",
                Minimum: "minimum",
            }
            if isinstance(config.fuzzy.t_norm.yager, float):
                if np.isclose(config.fuzzy.t_norm.yager, np.e):
                    w_parameter = "euler"
                elif np.isclose(config.fuzzy.t_norm.yager, (1 + 5**0.5) / 2):
                    w_parameter = "golden"
                else:
                    w_parameter = config.fuzzy.t_norm.yager
            else:
                w_parameter = config.fuzzy.t_norm.yager
            config.fuzzy.t_norm.yager = w_parameter

            if config.fuzzy.inference.t_norm in values_to_str:
                config.fuzzy.inference.t_norm = values_to_str[
                    config.fuzzy.inference.t_norm
                ]
        else:
            # map string values to their true values
            str_to_values = {
                "euler": np.e,
                "golden": (1 + 5**0.5) / 2,
                "algebraic_product": AlgebraicProduct,
                "minimum": Minimum,
            }
            if (
                isinstance(config.fuzzy.t_norm.yager, str)
                and config.fuzzy.t_norm.yager.lower() in str_to_values
            ):
                w_parameter: float = str_to_values[config.fuzzy.t_norm.yager.lower()]
            else:
                w_parameter: float = float(config.fuzzy.t_norm.yager)
            config.fuzzy.t_norm.yager = w_parameter

            if config.fuzzy.inference.t_norm in str_to_values:
                config.fuzzy.inference.t_norm = str_to_values[
                    config.fuzzy.inference.t_norm
                ]

    return config


def load_and_override_default_configuration(path: pathlib.Path) -> Config:
    """
    Load the default configuration file and override it with the configuration file given by
    'path'. This function is useful for when you want to override the default configuration
    settings with a configuration file that is not the default configuration file. For example,
    you may want to override the default configuration settings with the configuration settings
    for a specific experiment.

    Args:
        path: A file path to the configuration file that should be merged
        with the default configuration.

    Returns:
        The custom configuration settings.
    """
    # the default configuration
    configuration = load_configuration()
    # the custom configuration
    custom_configuration = load_configuration(path, convert_data_types=False)
    configuration.merge(custom_configuration, exclusive=False)
    # if configuration.output.verbose:
    #     configuration.print(ignored_keys=())
    return configuration
