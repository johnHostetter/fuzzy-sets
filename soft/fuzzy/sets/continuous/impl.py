"""
Implements various membership functions by inheriting from ContinuousFuzzySet.
"""
import pickle
import inspect
from pathlib import Path
from typing import List, NoReturn, Union, Tuple, Any, Dict, Set

import torch
from natsort import natsorted

from utilities.functions import convert_to_tensor, get_object_attributes
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

    def __init__(self, modules_list=None, expandable=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if modules_list is None:
            modules_list = []
        self.modules_list = torch.nn.ModuleList(modules_list)
        self.expandable = expandable
        self.epsilon = 0.5  # epsilon-completeness

        # self.neurons = torch.nn.ModuleList()
        # for idx in range(64):
        #     neuron = torch.nn.Sequential(
        #         torch.nn.Linear(
        #             in_features=49,
        #             out_features=128,
        #         ),
        #         torch.nn.LeakyReLU(),
        #         torch.nn.Linear(
        #             in_features=128,
        #             out_features=10,
        #         ),
        #         torch.nn.ReLU()
        #         # torch.nn.Sigmoid(),
        #     )
        #     self.neurons.add_module(f"{idx}", neuron)

    def __getattribute__(self, item):
        try:
            if item in ("centers", "widths", "sigmas"):
                modules_list = self.__dict__["_modules"]["modules_list"]
                if len(modules_list) > 0:
                    module_attributes: List[
                        torch.Tensor
                    ] = []  # the secondary response denoting module filter
                    for module in modules_list:
                        module_attributes.append(getattr(module, item))
                    return torch.cat(module_attributes, dim=-1)
                raise ValueError(
                    "The torch.nn.ModuleList of GroupedFuzzySets is empty."
                )
            return object.__getattribute__(self, item)
        except AttributeError:
            return self.__getattr__(item)

    def get_mask(self) -> Union[torch.Tensor, NoReturn]:
        if len(self.modules_list) > 0:
            module_masks: List[
                torch.Tensor
            ] = []  # the secondary response denoting module filter
            for module in self.modules_list:
                module_masks.append(module.get_mask().float())
            return torch.cat(module_masks, dim=-1)
        raise ValueError("The torch.nn.ModuleList of GroupedFuzzySets is empty.")

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
        if len(self.modules_list) > 0:
            # modules' responses are membership degrees when modules are ContinuousFuzzySet
            module_memberships: List[
                torch.Tensor
            ] = []  # the primary response from the module
            module_masks: List[
                torch.Tensor
            ] = []  # the secondary response denoting module filter
            for module in self.modules_list:
                membership: Membership = module(observations)
                module_memberships.append(membership.degrees)
                module_masks.append(membership.mask.float())
            return Membership(
                degrees=torch.cat(module_memberships, dim=-1),
                mask=torch.cat(module_masks, dim=-1),
            )
        raise ValueError("The torch.nn.ModuleList of GroupedFuzzySets is empty.")

    def forward(self, observations) -> Membership:
        observations = convert_to_tensor(observations)

        # memberships = []
        # for idx, neuron in enumerate(self.neurons):
        #     memberships.append(
        #         neuron(observations[:, idx, :, :].flatten(start_dim=1)).unsqueeze(dim=1)
        #     )
        # return Membership(
        #     degrees=torch.cat(memberships, dim=1),
        #     mask=torch.zeros((len(self.neurons), 10)),
        # )

        module_responses, module_masks = self.calculate_module_responses(observations)

        # if mu.grad_fn is None and self.expandable:
        #     print("grad_fn of mu is None, but might be in target net tho")

        if self.expandable and False:
            # find where the new centers should be added, if any
            new_centers = (
                observations * (module_responses.max(dim=-1).values <= self.epsilon)
            ) + ((module_responses.max(dim=-1).values > self.epsilon) / 0).nan_to_num(
                0.0, posinf=torch.nan
            )
            # print(mu.max(dim=-1).values, observations)
            nc: List[torch.Tensor] = [
                torch.tensor(
                    list(
                        set(
                            new_centers[:, var_idx][
                                ~new_centers[:, var_idx].isnan()
                            ].tolist()
                        )
                    )
                )
                for var_idx in range(new_centers.shape[-1])
            ]

            # doing some trick w. dynamically expandable networks
            sets: List[Any] = []  # 'Any' as in *any* child/impl. of ContinuousFuzzySet
            empty_sets: List[int] = []
            for idx, n in enumerate(nc):
                if len(n) == 0:
                    empty_sets.append(idx)
                sets.append(
                    Gaussian(
                        centers=n,
                        widths=torch.randn_like(n).abs(),
                    )
                )
            if len(sets) > 0 and len(empty_sets) < len(sets):
                # take the fuzzy sets, and make a ContinuousFuzzySet efficient object
                g = ContinuousFuzzySet.stack(sets)
                new_idx = len(self.modules_list)
                print(
                    f"adding {g.centers.shape}; num of modules already: {len(self.modules_list)}"
                )
                print(f"to dimensions: {set(range(len(sets))) - set(empty_sets)}")
                self.modules_list.add_module(str(new_idx), g)
                module_responses, module_masks = self.calculate_module_responses(
                    observations
                )

        return Membership(degrees=module_responses, mask=module_masks)


# class Neuron(ContinuousFuzzySet):
#     def __init__(self, in_features, observations):
#         super().__init__(in_features)
#         self.neurons = torch.nn.ModuleList()
#         for idx in range(observations.shape[1]):
#             neuron = torch.nn.Sequential(
#                 torch.nn.Linear(
#                     in_features=observations.flatten(start_dim=2).shape[-1],
#                     out_features=10,
#                 ),
#                 torch.nn.Sigmoid(),
#             )
#             self.neurons.add_module(f"{idx}", neuron)
#
#     def calculate_membership(
#         self, observations: torch.Tensor
#     ) -> Union[NoReturn, torch.Tensor]:
#         memberships = []
#         for idx, neuron in enumerate(self.neurons):
#             memberships.append(neuron(observations[:, idx, :, :]))
#         return Membership(
#             degrees=torch.cat(memberships, dim=1),
#             mask=torch.zeros((len(self.neurons), self.neurons[0].in_features)),
#         )


class Gaussian(ContinuousFuzzySet):
    """
    Implementation of the Gaussian membership function, written in PyTorch.
    """

    def __init__(
        self,
        centers=None,
        widths=None,
        labels: List[str] = None,
    ):
        super().__init__(centers=centers, widths=widths, labels=labels)

    @property
    def sigmas(self) -> torch.Tensor:
        """
        Gets the sigma for the Gaussian fuzzy set; alias for the 'widths' parameter.

        Returns:
            torch.Tensor
        """
        return self.widths

    @sigmas.setter
    def sigmas(self, sigmas) -> None:
        """
        Sets the sigma for the Gaussian fuzzy set; alias for the 'widths' parameter.

        Returns:
            None
        """
        self.widths = sigmas

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        # observations = torch.nn.functional.avg_pool2d(
        #     observations, kernel_size=observations.size()[2:]
        # ).view(observations.size()[0], -1)

        # sim = torch.nn.CosineSimilarity(dim=-1)
        # return sim(
        #     observations.flatten(start_dim=2).unsqueeze(2),
        #     self.centers.flatten(start_dim=2),
        # )
        #
        # SSD = (
        #     torch.pow((observations.unsqueeze(dim=2) - self.centers), 2).sum(-1).sum(-1)
        # )
        #
        # return SSD / torch.pow(self.centers, 2).sum(-1).sum(-1)  # normalized SSD
        #
        # try:
        #     return (
        #         torch.exp(
        #             -1.0
        #             * (
        #                 torch.pow(
        #                     observations - self.centers,
        #                     2,
        #                 )
        #                 / (torch.pow(self.widths, 2) + 1e-32)
        #             )
        #         )
        #         * self.mask
        #     )
        # except RuntimeError:  # computer vision scenario
        #     # return torch.nn.CosineSimilarity(dim=-1)(
        #     #     observations.view(observations.shape[0], -1).unsqueeze(dim=1),
        #     #     self.centers.view(self.centers.shape[0], -1),
        #     # ).unsqueeze(
        #     #     dim=-1
        #     # )  # need a placeholder for the term slot
        #     # observations = torch.nn.functional.avg_pool2d(
        #     #     observations, kernel_size=observations.size()[2:]
        #     # ).view(observations.size()[0], -1)
        #
        #     # return torch.nn.CosineSimilarity(dim=-1)(
        #     #     observations.view(*observations.shape[:2], -1).unsqueeze(dim=1),
        #     #     self.centers.view(*self.centers.shape[:-2], -1),
        #     # )
        #     # observations = observations.to("cuda:0")
        #
        #     SSD = (
        #         torch.pow((observations.unsqueeze(dim=2) - self.centers), 2)
        #         .sum(-1)
        #         .sum(-1)
        #     )
        #
        #     return SSD / torch.pow(self.centers, 2).sum(-1).sum(-1)  # normalized SSD
        #
        #     return torch.pow(
        #         observations.flatten(start_dim=1).unsqueeze(dim=-1) - self.centers,
        #         2,
        #     ) / (torch.pow(self.widths, 2) + 1e-32)
        #
        #     print()
        #
        #     return (
        #         torch.exp(
        #             -1.0
        #             * (
        #                 (
        #                     torch.pow(
        #                         observations.unsqueeze(dim=-1).cuda()
        #                         - self.centers[:, 0, :].cuda(),
        #                         2,
        #                     )
        #                     / (torch.pow(self.widths[:, 0, :].cuda(), 2) + 1e-32)
        #                 )
        #                 + (
        #                     torch.pow(
        #                         observations.unsqueeze(dim=-1).cuda()
        #                         - self.centers[:, 1, :].cuda(),
        #                         2,
        #                     )
        #                     / (torch.pow(self.widths[:, 1, :].cuda(), 2) + 1e-32)
        #                 )
        #             )
        #         )
        #         * (1 / 2 * torch.pi * self.widths.cuda().prod(dim=1))
        #         # * self.mask.cuda()
        #         # .unsqueeze(dim=0)
        #         # .transpose(1, 2)
        #     )

        return (
            torch.exp(
                -1.0
                * (
                    torch.pow(
                        observations - self.centers,
                        2,
                    )
                    / (torch.pow(self.widths, 2) + 1e-32)
                )
            )
            * self.mask
            # .unsqueeze(dim=0)
            # .transpose(1, 2)
        )

        # return (
        #     torch.exp(
        #         -1.0
        #         * (
        #             torch.pow(
        #                 observations.unsqueeze(1) - self.centers,
        #                 2,
        #             )
        #             / (torch.pow(self.widths[None, :, :, None, None], 2) + 1e-32)
        #         )
        #     )
        #     * self.mask[None, :, :, None, None]
        # )
        # torch.nn.CosineSimilarity()(
        #     observations.view(observations.shape[0], -1).unsqueeze(dim=1),
        #     self.centers.view(self.centers.shape[0], -1),
        # )


#
#
# torch.nn.CosineSimilarity(dim=0)(
#     observations.view(observations.shape[0], -1)[0],
#     self.centers.view(self.centers.shape[0], -1)[0],
# )
# torch.nn.CosineSimilarity(dim=-1)(
#     observations.view(observations.shape[0], -1).unsqueeze(dim=1),
#     self.centers.view(self.centers.shape[0], -1),
# )


class Lorentzian(ContinuousFuzzySet):
    """
    Implementation of the Lorentzian membership function, written in PyTorch.
    """

    def __init__(
        self,
        centers=None,
        widths=None,
        labels: List[str] = None,
    ):
        super().__init__(centers=centers, widths=widths, labels=labels)

    @property
    def sigmas(self) -> torch.Tensor:
        """
        Gets the sigma for the Lorentzian fuzzy set; alias for the 'widths' parameter.

        Returns:
            torch.Tensor
        """
        return self.widths

    @sigmas.setter
    def sigmas(self, sigmas) -> None:
        """
        Sets the sigma for the Lorentzian fuzzy set; alias for the 'widths' parameter.

        Returns:
            None
        """
        self.widths = sigmas

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mask * (
            1
            / (
                torch.pow(
                    (self.centers - observations.unsqueeze(dim=-1))
                    / (0.5 * self.widths),
                    2,
                )
                + 1
            )
        )


class LogisticCurve(torch.nn.Module):
    """
    A generic torch.nn.Module class that implements a logistic curve, which allows us to
    tune the midpoint, and growth of the curve, with a fixed supremum (the supremum is
    the maximum value of the curve).
    """

    def __init__(self, midpoint, growth, supremum):
        super().__init__()
        self.midpoint = torch.nn.parameter.Parameter(
            convert_to_tensor(midpoint)
        ).float()
        self.growth = torch.nn.parameter.Parameter(
            convert_to_tensor(growth).double()
        ).float()
        self.supremum = convert_to_tensor(
            supremum  # not a parameter, so we don't want to track it
        ).float()

    def forward(self, tensors: torch.Tensor) -> torch.Tensor:
        """
        Calculate the value of the logistic curve at the given point.

        Args:
            tensors:

        Returns:

        """
        return self.supremum / (1 + torch.exp(-self.growth * (tensors - self.midpoint)))


class Triangular(ContinuousFuzzySet):
    """
    Implementation of the Triangular membership function, written in PyTorch.
    """

    def __init__(
        self,
        centers=None,
        widths=None,
        labels: List[str] = None,
    ):
        super().__init__(centers=centers, widths=widths, labels=labels)

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the function. Applies the function to the input elementwise.

        Args:
            observations: Two-dimensional matrix of observations,
            where a row is a single observation and each column
            is related to an attribute measured during that observation.

        Returns:
            The membership degrees of the observations for the Triangular fuzzy set.
        """
        return (
            torch.max(
                1.0 - (1.0 / self.widths) * torch.abs(observations - self.centers),
                torch.tensor(0.0),
            )
            * self.mask
        )
