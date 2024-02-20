"""
Utility functions, such as for getting the powerset of an iterable.
"""
import inspect
from typing import Dict, Any, List, Set
from collections.abc import Iterable
from itertools import chain, combinations

import torch
import numpy as np


def powerset(iterable: Iterable, min_items: int):
    """
    Get the powerset of an iterable.

    Args:
        iterable: An iterable collection of elements.
        min_items: The minimum number of items that must be in each subset.

    Returns:
        The powerset of the given iterable.
    """
    # https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    return chain.from_iterable(
        combinations(list(iterable), r)
        for r in range(min_items, len(list(iterable)) + 1)
    )


def all_subclasses(cls) -> Set[Any]:
    """
    Get all subclasses of the given class, recursively.

    Returns:
        A set of all subclasses of the given class.
    """
    return {cls}.union(s for c in cls.__subclasses__() for s in all_subclasses(c))


def find_centers_and_widths(
    data_point, minimums, maximums, alpha: float
) -> List[Dict[str, float]]:
    """
    Find the centers and widths to be used for a newly created fuzzy set.

    Args:
        data_point (1D Numpy array): A single input_data observation where
            each column is a feature/attribute.
        minimums (iterable): The minimum value per feature in X.
        maximums (iterable): The maximum value per feature in X.
        alpha (float): A hyperparameter to adjust the generated widths' coverage.

    Returns:
        A list of dictionaries, where each dictionary contains the center and width
        for a newly created fuzzy set (that is to be created later).
    """
    # The variable 'theta' is added to accommodate for the instance in which an observation has
    # values that are the minimum/maximum. Otherwise, when determining the Gaussian membership,
    # a division by zero will occur; it essentially acts as an error tolerance.
    theta: float = 1e-8
    parameters: List[Dict[str, float]] = []
    for dim, attribute_value in enumerate(data_point):
        left_width: float = torch.sqrt(
            -1.0
            * (torch.pow((minimums[dim] - attribute_value) + theta, 2) / np.log(alpha))
        ).item()
        right_width: float = torch.sqrt(
            -1.0
            * (torch.pow((maximums[dim] - attribute_value) + theta, 2) / np.log(alpha))
        ).item()
        aggregated_sigma: float = regulator(left_width, right_width)
        parameters.append({"centers": attribute_value, "widths": aggregated_sigma})
    return parameters


def regulator(sigma_1: float, sigma_2: float) -> float:
    """
    Regulator function as defined in CLIP.

    Args:
        sigma_1: The left sigma/width.
        sigma_2: The right sigma/width.

    Returns:
        sigma (float): An adjusted sigma so that the produced
        Gaussian membership function is not warped.
    """
    return (1 / 2) * (sigma_1 + sigma_2)


def convert_to_tensor(values: np.ndarray) -> torch.Tensor:
    """
    If the given values are not torch.Tensor, convert them to torch.Tensor.

    Args:
        values: Values such as the centers or widths of a fuzzy set.

    Returns:
        torch.tensor(np.array(values))
    """
    if isinstance(values, torch.Tensor):
        return values
    return torch.tensor(np.array(values)).float()


def get_object_attributes(obj_instance) -> Dict[str, Any]:
    """
    Get the attributes of an object instance.
    """
    # get the attributes that are local to the class, but may be inherited from the super class
    local_attributes = inspect.getmembers(
        obj_instance,
        lambda attr: not (inspect.ismethod(attr)) and not (inspect.isfunction(attr)),
    )
    # get the attributes that are inherited from (or found within) the super class
    super_attributes = inspect.getmembers(
        obj_instance.__class__.__bases__[0],
        lambda attr: not (inspect.ismethod(attr)) and not (inspect.isfunction(attr)),
    )
    # get the attributes that are local to the class, but not inherited from the super class
    return {
        attr: value
        for attr, value in local_attributes
        if (attr, value) not in super_attributes and not attr.startswith("_")
    }


class GaussianDropout(torch.nn.Module):
    """
    Gaussian Dropout as defined in the paper "Gaussian Dropout" by Srivastava et al. (2014).

    Similar to:
        https://keras.io/api/layers/regularization_layers/gaussian_dropout/
    """

    def __init__(self, probability=0.5):
        """
        Initialize the Gaussian Dropout layer.
        """
        super().__init__()
        if probability <= 0 or probability >= 1:
            raise ValueError("Probability value, p, should accomplish 0 < p < 1")
        self.probability = probability

    def forward(self, tensor: torch.Tensor):
        """
        Apply Gaussian Dropout to the input tensor.
        """
        if self.training:
            standard_deviation = (self.probability / (1.0 - self.probability)) ** 0.5
            epsilon = torch.rand_like(tensor) * standard_deviation
            return tensor * epsilon
        return tensor


def raw_dropout(tensor: torch.Tensor, probability):
    """
    Apply raw dropout to the input tensor.
    """
    # generate a binary mask based on the dropout probability
    shape: list = list(tensor.shape)
    shape[-1] = 2

    weights = torch.empty(shape, dtype=torch.float)
    weights[:, :, 0] = probability
    weights[:, :, 1] = 1 - probability
    mask = torch.multinomial(
        weights.view(-1, 2),
        num_samples=tensor.shape[-1],
        replacement=True,
    ).view(tensor.shape)

    # apply the mask to the input tensor
    return tensor * mask  # my defn, weight balancing
    # return (tensor * mask) / (1 - probability)
