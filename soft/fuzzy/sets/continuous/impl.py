"""
Implements various membership functions by inheriting from ContinuousFuzzySet.
"""
from typing import List, NoReturn, Union, Tuple, Any

import torch

from utilities.functions import convert_to_tensor
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

    def __init__(self, modules=None, expandable=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if modules is None:
            modules = []
        self.modules_list = torch.nn.ModuleList(modules)
        self.expandable = expandable
        self.epsilon = 0.5  # epsilon-completeness

        self.neurons = torch.nn.ModuleList()
        for idx in range(64):
            neuron = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=49,
                    out_features=128,
                ),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    in_features=128,
                    out_features=10,
                ),
                torch.nn.ReLU()
                # torch.nn.Sigmoid(),
            )
            self.neurons.add_module(f"{idx}", neuron)

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

    def calculate_module_responses(
        self, observations
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], NoReturn]:
        if len(self.modules_list) > 0:
            # modules' responses are membership degrees when modules are ContinuousFuzzySet
            module_responses: List[
                torch.Tensor
            ] = []  # the primary response from the module
            module_masks: List[
                torch.Tensor
            ] = []  # the secondary response denoting module filter
            for module in self.modules_list:
                module_response, module_mask = module(observations)
                module_responses.append(module_response)
                module_masks.append(module_mask.float())
            return torch.cat(module_responses, dim=-1), torch.cat(module_masks, dim=-1)
        raise ValueError("The torch.nn.ModuleList of GroupedFuzzySets is empty.")

    def forward(self, observations) -> Membership:
        observations = convert_to_tensor(observations)

        memberships = []
        for idx, neuron in enumerate(self.neurons):
            memberships.append(
                neuron(observations[:, idx, :, :].flatten(start_dim=1)).unsqueeze(dim=1)
            )
        return Membership(
            degrees=torch.cat(memberships, dim=1),
            mask=torch.zeros((len(self.neurons), 10)),
        )

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
                        in_features=len(n),
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


class Neuron(ContinuousFuzzySet):
    def __init__(self, in_features, observations):
        super().__init__(in_features)
        self.neurons = torch.nn.ModuleList()
        for idx in range(observations.shape[1]):
            neuron = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=observations.flatten(start_dim=2).shape[-1],
                    out_features=10,
                ),
                torch.nn.Sigmoid(),
            )
            self.neurons.add_module(f"{idx}", neuron)

    def calculate_membership(
        self, observations: torch.Tensor
    ) -> Union[NoReturn, torch.Tensor]:
        memberships = []
        for idx, neuron in enumerate(self.neurons):
            memberships.append(neuron(observations[:, idx, :, :]))
        return Membership(
            degrees=torch.cat(memberships, dim=1),
            mask=torch.zeros((len(self.neurons), self.neurons[0].in_features)),
        )


class Gaussian(ContinuousFuzzySet):
    """
    Implementation of the Gaussian membership function, written in PyTorch.
    """

    def __init__(self, in_features, centers=None, widths=None, labels=None):
        super().__init__(in_features, centers=centers, widths=widths, labels=labels)

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
        if observations.ndim < self.centers.ndim:
            observations = observations.unsqueeze(dim=-1)

        # observations = torch.nn.functional.avg_pool2d(
        #     observations, kernel_size=observations.size()[2:]
        # ).view(observations.size()[0], -1)

        sim = torch.nn.CosineSimilarity(dim=-1)
        return sim(
            observations.flatten(start_dim=2).unsqueeze(2),
            self.centers.flatten(start_dim=2),
        )

        SSD = (
            torch.pow((observations.unsqueeze(dim=2) - self.centers), 2).sum(-1).sum(-1)
        )

        return SSD / torch.pow(self.centers, 2).sum(-1).sum(-1)  # normalized SSD

        try:
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
            )
        except RuntimeError:  # computer vision scenario
            # return torch.nn.CosineSimilarity(dim=-1)(
            #     observations.view(observations.shape[0], -1).unsqueeze(dim=1),
            #     self.centers.view(self.centers.shape[0], -1),
            # ).unsqueeze(
            #     dim=-1
            # )  # need a placeholder for the term slot
            # observations = torch.nn.functional.avg_pool2d(
            #     observations, kernel_size=observations.size()[2:]
            # ).view(observations.size()[0], -1)

            # return torch.nn.CosineSimilarity(dim=-1)(
            #     observations.view(*observations.shape[:2], -1).unsqueeze(dim=1),
            #     self.centers.view(*self.centers.shape[:-2], -1),
            # )
            # observations = observations.to("cuda:0")

            SSD = (
                torch.pow((observations.unsqueeze(dim=2) - self.centers), 2)
                .sum(-1)
                .sum(-1)
            )

            return SSD / torch.pow(self.centers, 2).sum(-1).sum(-1)  # normalized SSD

            return torch.pow(
                observations.flatten(start_dim=1).unsqueeze(dim=-1) - self.centers,
                2,
            ) / (torch.pow(self.widths, 2) + 1e-32)

            print()

            return (
                torch.exp(
                    -1.0
                    * (
                        (
                            torch.pow(
                                observations.unsqueeze(dim=-1).cuda()
                                - self.centers[:, 0, :].cuda(),
                                2,
                            )
                            / (torch.pow(self.widths[:, 0, :].cuda(), 2) + 1e-32)
                        )
                        + (
                            torch.pow(
                                observations.unsqueeze(dim=-1).cuda()
                                - self.centers[:, 1, :].cuda(),
                                2,
                            )
                            / (torch.pow(self.widths[:, 1, :].cuda(), 2) + 1e-32)
                        )
                    )
                )
                * (1 / 2 * torch.pi * self.widths.cuda().prod(dim=1))
                # * self.mask.cuda()
                # .unsqueeze(dim=0)
                # .transpose(1, 2)
            )

            # return (
            #     torch.exp(
            #         -1.0
            #         * (
            #             torch.pow(
            #                 observations.unsqueeze(dim=-1).cuda() - self.centers.cuda(),
            #                 2,
            #             )
            #             / (torch.pow(self.widths.cuda(), 2) + 1e-32)
            #         )
            #     )
            #     * self.mask.cuda()
            #     # .unsqueeze(dim=0)
            #     # .transpose(1, 2)
            # )

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

    def __init__(self, in_features, centers=None, widths=None, labels=None):
        super().__init__(in_features, centers=centers, widths=widths, labels=labels)

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
        return (1 - self.get_mask()).to(observations.device) * (
            1
            / (
                torch.pow(
                    (
                        self.centers.to(observations.device)
                        - observations.unsqueeze(dim=-1)
                    )
                    / (0.5 * self.widths.to(observations.device)),
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

    def __init__(self, in_features, centers=None, widths=None, labels=None):
        super().__init__(in_features, centers=centers, widths=widths, labels=labels)

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
                1.0
                - (1.0 / self.widths)
                * torch.abs(observations.unsqueeze(dim=-1) - self.centers),
                torch.tensor(0.0),
            )
            * self.mask
        )
