"""
This module contains the GroupedFuzzySets class, which is a generic and abstract torch.nn.Module
class that contains a torch.nn.ModuleList of ContinuousFuzzySet objects. The expectation here is
that each ContinuousFuzzySet may define fuzzy sets of different conventions, such as Gaussian,
Triangular, Trapezoidal, etc. Then, subsequent inference engines can handle these heterogeneously
defined fuzzy sets with no difficulty. Further, this class was specifically designed to incorporate
dynamic addition of new fuzzy sets in the construction of neuro-fuzzy networks via network morphism.
"""

import pickle
import inspect
from pathlib import Path
from typing import List, NoReturn, Union, Tuple, Any, Dict, Set, Type

import torch
import numpy as np
from natsort import natsorted
from scipy.signal import find_peaks

from soft.fuzzy.sets.continuous.impl import LogGaussian, Gaussian
from soft.utilities.functions import (
    convert_to_tensor,
    get_object_attributes,
    find_centers_and_widths,
)
from soft.fuzzy.sets.continuous.abstract import ContinuousFuzzySet, Membership


class GroupedFuzzySets(torch.nn.Module):
    """
    A generic and abstract torch.nn.Module class that contains a torch.nn.ModuleList
    of ContinuousFuzzySet objects. The expectation here is that each ContinuousFuzzySet
    may define fuzzy sets of different conventions, such as Gaussian, Triangular, Trapezoidal, etc.
    Then, subsequent inference engines can handle these heterogeneously defined fuzzy sets
    with no difficulty. Further, this class was specifically designed to incorporate dynamic
    addition of new fuzzy sets in the construction of neuro-fuzzy networks via network morphism.

    However, this class does *not* carry out any functionality that is necessarily tied to fuzzy
    sets, it is simply named so as this was its intended purpose - grouping fuzzy sets. In other
    words, the same "trick" of using a torch.nn.ModuleList of torch.nn.Module objects applies to
    any kind of torch.nn.Module object.
    """

    def __init__(self, *args, modules_list=None, expandable=False, **kwargs):
        super().__init__(*args, **kwargs)
        if modules_list is None:
            modules_list = []
        self.modules_list = torch.nn.ModuleList(modules_list)
        self.expandable = expandable
        self.pruning = False
        self.epsilon = 0.1  # epsilon-completeness
        # keep track of minimums and maximums if for fuzzy set width calculation
        self.domain: Dict[str, Union[None, torch.Tensor]] = {
            "minimums": None,
            "maximums": None,
        }
        # store data that we have seen to later add new fuzzy sets
        self.data_seen: List[torch.Tensor] = []
        # after we see this many data points, we will update the fuzzy sets
        self.data_limit_until_update: int = 64

    def __getattribute__(self, item):
        try:
            if item in ("centers", "widths", "sigmas"):
                modules_list = self.__dict__["_modules"]["modules_list"]
                if len(modules_list) > 0:
                    module_attributes: List[torch.Tensor] = (
                        []
                    )  # the secondary response denoting module filter
                    for module in modules_list:
                        module_attributes.append(getattr(module, item))
                    return torch.cat(module_attributes, dim=-1)
                raise ValueError(
                    "The torch.nn.ModuleList of GroupedFuzzySets is empty."
                )
            return object.__getattribute__(self, item)
        except AttributeError:
            return self.__getattr__(item)

    def save(self, path: Path) -> None:
        """
        Save the model to the given path.

        Args:
            path: The path to save the GroupedFuzzySet to.

        Returns:
            None
        """
        # get the attributes that are local to the class, but not inherited from the super class
        local_attributes_only = get_object_attributes(self)

        # save a reference to the attributes (and their values) so that when iterating over them,
        # we do not modify the dictionary while iterating over it (which would cause an error)
        # we modify the dictionary by removing attributes that have a value of torch.nn.ModuleList
        # because we want to save the modules in the torch.nn.ModuleList separately
        local_attributes_only_items: List[Tuple[str, Any]] = list(
            local_attributes_only.items()
        )
        for attr, value in local_attributes_only_items:
            if isinstance(
                value, torch.nn.ModuleList
            ):  # e.g., attr may be self.modules_list
                for idx, module in enumerate(value):
                    subdirectory = path / attr / str(idx)
                    subdirectory.mkdir(parents=True, exist_ok=True)
                    if isinstance(module, ContinuousFuzzySet):
                        # save the fuzzy set using the fuzzy set's special protocol
                        module.save(
                            path / attr / str(idx) / f"{module.__class__.__name__}.pt"
                        )
                    else:
                        # unknown and unrecognized module, but attempt to save the module
                        torch.save(
                            module,
                            path / attr / str(idx) / f"{module.__class__.__name__}.pt",
                        )
                # remove the torch.nn.ModuleList from the local attributes
                del local_attributes_only[attr]

        # save the remaining attributes
        with open(path / f"{self.__class__.__name__}.pickle", "wb") as handle:
            pickle.dump(local_attributes_only, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "GroupedFuzzySets":
        """
        Load the model from the given path.

        Args:
            path: The path to load the GroupedFuzzySet from.

        Returns:
            The loaded GroupedFuzzySet.
        """
        modules_list = []
        local_attributes_only: Dict[str, Any] = {}
        for file_path in path.iterdir():
            if ".pickle" in file_path.name:
                # load the remaining attributes
                with open(file_path, "rb") as handle:
                    local_attributes_only.update(pickle.load(handle))
            elif file_path.is_dir():
                for subdirectory in natsorted(file_path.iterdir()):
                    if subdirectory.is_dir():
                        module_path: Path = list(subdirectory.glob("*.pt"))[0]
                        # load the fuzzy set using the fuzzy set's special protocol
                        class_name: str = module_path.name.split(".pt")[0]
                        try:
                            modules_list.append(
                                ContinuousFuzzySet.get_subclass(class_name).load(
                                    module_path
                                )
                            )
                        except ValueError:
                            # unknown and unrecognized module, but attempt to load the module
                            modules_list.append(torch.load(module_path))
                    else:
                        raise UserWarning(
                            f"Unexpected file found in {file_path}: {module_path}"
                        )
                local_attributes_only[file_path.name] = modules_list

        # of the remaining attributes, we must determine which are shared between the
        # super class and the local class, otherwise we will get an error when trying to
        # initialize the local class (more specifically, the torch.nn.Module __init__ method
        # requires self.call_super_init to be set to True, but then the attribute would exist
        # as a super class attribute, and not a local class attribute)
        shared_args: Set[str] = set(
            inspect.signature(cls).parameters.keys()
        ).intersection(local_attributes_only.keys())

        # create the GroupedFuzzySet object with the shared arguments
        # (e.g., modules_list, expandable)
        grouped_fuzzy_set: GroupedFuzzySets = cls(
            **{
                key: value
                for key, value in local_attributes_only.items()
                if key in shared_args
            }
        )

        # determine the remaining attributes
        remaining_args: Dict[str, Any] = {
            key: value
            for key, value in local_attributes_only.items()
            if key not in shared_args
        }

        # set the remaining attributes
        for attr, value in remaining_args.items():
            setattr(grouped_fuzzy_set, attr, value)

        return grouped_fuzzy_set

    def calculate_module_responses(
        self, observations
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], NoReturn]:
        """
        Calculate the responses from the modules in the torch.nn.ModuleList of GroupedFuzzySets.
        """
        if len(self.modules_list) > 0:
            # modules' responses are membership degrees when modules are ContinuousFuzzySet
            module_elements: List[torch.Tensor] = []
            module_memberships: List[torch.Tensor] = (
                []
            )  # the primary response from the module
            module_masks: List[torch.Tensor] = (
                []
            )  # the secondary response denoting module filter
            for module in self.modules_list:
                membership: Membership = module(observations)
                module_elements.append(membership.elements.cpu())
                module_memberships.append(membership.degrees.cpu())
                module_masks.append(membership.mask.float().cpu())
            return Membership(
                elements=torch.cat(module_elements, dim=-1).to(observations.device),
                degrees=torch.cat(module_memberships, dim=-1).to(observations.device),
                mask=torch.cat(module_masks, dim=-1).to(observations.device),
            )
        raise ValueError("The torch.nn.ModuleList of GroupedFuzzySets is empty.")

    def expand(
        self, observations, module_responses, module_masks
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expand the GroupedFuzzySets if necessary.
        """
        if self.expandable and self.training:
            # save the data that we have seen
            self.data_seen.append(observations)

            # keep a running tally of mins and maxs of the domain
            minimums = observations.min(dim=0).values
            maximums = observations.max(dim=0).values
            self.domain["minimums"] = (
                minimums
                if self.domain["minimums"] is None
                else torch.min(self.domain["minimums"], maximums)
            )
            self.domain["maximums"] = (
                maximums
                if self.domain["maximums"] is None
                else torch.max(self.domain["maximums"], maximums)
            )

            if len(self.data_seen) % self.data_limit_until_update == 0:
                # find where the new centers should be added, if any
                # LogGaussian was used, then use following to check for real membership degrees:
                for module in self.modules_list:
                    if isinstance(module, LogGaussian) and not isinstance(
                        module, Gaussian
                    ):
                        with torch.no_grad():
                            assert (
                                module_responses.exp() * module_masks
                            ).max().item() <= 1.0

                all_data = torch.vstack(self.data_seen)
                exemplars: List[torch.Tensor] = []

                for var_idx in range(all_data.shape[-1]):
                    exemplars.append(
                        torch.Tensor(
                            self.evenly_spaced_exemplars(
                                all_data[:, var_idx].detach().cpu().numpy(), 3
                            )
                        )
                    )
                    if len(exemplars[-1]) == 0:
                        exemplars = (
                            []
                        )  # discard everything, no exemplars found in this dimension
                        break  # stop the loop, no need to continue
                if len(exemplars) == 0:
                    # no exemplars found in any dimension
                    return observations, module_responses, module_masks

                try:
                    exemplars: torch.Tensor = torch.hstack(exemplars)
                except RuntimeError:
                    # no exemplars found in any dimension
                    return observations, module_responses, module_masks
                # Create a new matrix with nan values
                new_centers = torch.full_like(exemplars, float("nan"))

                # Use torch.where to update values that satisfy the condition
                new_centers = torch.where(
                    self.calculate_module_responses(exemplars)
                    .degrees.max(dim=-1)
                    .values
                    < self.epsilon,
                    exemplars,
                    new_centers,
                )

                if not new_centers.isnan().all():  # add new centers
                    terms: List[Dict[str, float]] = find_centers_and_widths(
                        data_point=new_centers.nan_to_num(0).mean(dim=0),
                        minimums=self.domain["minimums"],
                        maximums=self.domain["maximums"],
                        alpha=0.3,
                    )

                    new_widths = torch.Tensor([term["widths"] for term in terms])

                    # assert new_widths.isnan().any() is False

                    # create the widths for the new centers
                    new_widths = (
                        # only keep the widths for the entries that are not torch.nan
                        ~torch.isnan(new_centers)
                        * new_widths
                    ) + (torch.isnan(new_centers) * -1)

                    # above result is tensor that contains new centers, but also contains torch.nan
                    # in the places where a new center is not needed

                    module_type = type(self.modules_list[0])
                    if issubclass(module_type, ContinuousFuzzySet):
                        granule = ContinuousFuzzySet.get_subclass(module_type.__name__)(
                            centers=new_centers.nan_to_num(0.0)
                            .transpose(0, 1)
                            .max(dim=-1, keepdim=True)
                            .values,
                            widths=new_widths.transpose(0, 1)
                            .max(dim=-1, keepdim=True)
                            .values,
                        )
                    else:
                        raise ValueError(
                            "The module type is not ContinuousFuzzySet, and therefore cannot "
                            "be used for dynamic expansion."
                        )
                    print(
                        f"add {granule.centers.shape}; modules already: {len(self.modules_list)}"
                    )
                    # print(f"to dimensions: {set(range(len(sets))) - set(empty_sets)}")
                    self.modules_list.add_module(str(len(self.modules_list)), granule)

                    # clear the history
                    self.data_seen = []

                    # reduce the number of torch.nn.Modules in the list for computational efficiency
                    # (this is not necessary, but it is a good idea)
                    self.prune(module_type)

            (
                _,
                module_responses,
                module_masks,
            ) = self.calculate_module_responses(observations)
        return observations, module_responses, module_masks

    @staticmethod
    def evenly_spaced_exemplars(data: np.ndarray, max_peaks: int) -> np.ndarray:
        """
        Find the peaks in the data and return the peaks, or a subset of the peaks if there are
        more than max_peaks.

        Args:
            data: The data to find the peaks in.
            max_peaks: The maximum number of peaks to return.

        Returns:
            The peaks, or a subset of the peaks if there are more than max_peaks.
        """
        peaks, _ = find_peaks(data)
        if len(peaks) <= max_peaks:
            return peaks

        sampled_peaks_indices = np.linspace(0, len(peaks) - 1, max_peaks).astype(int)
        sampled_peaks = peaks[sampled_peaks_indices]
        return data[sampled_peaks][:, None]

    def prune(self, module_type: Type[ContinuousFuzzySet]) -> None:
        """
        Prune the torch.nn.ModuleList of GroupedFuzzySets by keeping the first module, but
        collapsing the rest of the modules into a single module. This is done to reduce the
        number of torch.nn.Modules in the list for computational efficiency.
        """
        if self.pruning and len(self.modules_list) > 5:
            centers, widths = [], []
            for module in self.modules_list[1:]:
                if module.centers.shape[-1] > 1:
                    centers.append(module.centers.mean(dim=-1, keepdim=True))
                    widths.append(module.widths.max(dim=-1, keepdim=True).values)
                else:
                    centers.append(module.centers)
                    widths.append(module.widths)
            if issubclass(module_type, ContinuousFuzzySet):
                module = ContinuousFuzzySet.get_subclass(module_type.__name__)(
                    centers=torch.cat(centers, dim=-1),
                    widths=torch.cat(widths, dim=-1),
                )
                print(module.centers.shape)
            else:
                raise ValueError(
                    "The module type is not ContinuousFuzzySet, and therefore cannot "
                    "be used for dynamic expansion."
                )
            self.modules_list = torch.nn.ModuleList([self.modules_list[0], module])

    def forward(self, observations) -> Membership:
        """
        Calculate the responses from the modules in the torch.nn.ModuleList of GroupedFuzzySets.
        Expand the GroupedFuzzySets if necessary.
        """
        observations = convert_to_tensor(observations)
        (
            _,  # module_elements
            module_responses,
            module_masks,
        ) = self.calculate_module_responses(observations)

        self.expand(observations, module_responses, module_masks)

        return Membership(
            elements=observations, degrees=module_responses, mask=module_masks
        )
