import shutil
import unittest
from pathlib import Path

import torch

from soft.fuzzy.sets.continuous.impl import Gaussian
from soft.fuzzy.sets.continuous.group import GroupedFuzzySets
from soft.utilities.functions import get_object_attributes


class TestGroupedFuzzySets(unittest.TestCase):
    """
    Test the GroupedFuzzySets class.
    """

    def test_save_grouped_fuzzy_sets(self):
        """
        Test saving grouped fuzzy sets.
        """
        grouped_fuzzy_sets: GroupedFuzzySets = GroupedFuzzySets(
            modules_list=[
                Gaussian.create(
                    number_of_variables=2,
                    number_of_terms=3,
                ),
                Gaussian.create(
                    number_of_variables=2,
                    number_of_terms=3,
                ),
            ]
        )
        grouped_fuzzy_sets.save(Path("test_grouped_fuzzy_sets"))
        loaded_grouped_fuzzy_sets: GroupedFuzzySets = grouped_fuzzy_sets.load(
            Path("test_grouped_fuzzy_sets")
        )

        for i in range(len(grouped_fuzzy_sets.modules_list)):
            assert torch.equal(
                grouped_fuzzy_sets.modules_list[i].centers,
                loaded_grouped_fuzzy_sets.modules_list[i].centers,
            )
            assert torch.equal(
                grouped_fuzzy_sets.modules_list[i].widths,
                loaded_grouped_fuzzy_sets.modules_list[i].widths,
            )
            assert torch.equal(
                grouped_fuzzy_sets.modules_list[i].sigmas,
                loaded_grouped_fuzzy_sets.modules_list[i].sigmas,
            )

        # check the remaining attributes are the same
        for attribute in get_object_attributes(grouped_fuzzy_sets):
            value = getattr(grouped_fuzzy_sets, attribute)
            if isinstance(value, torch.nn.ModuleList):
                continue  # already checked above
            assert value == getattr(loaded_grouped_fuzzy_sets, attribute)

        # delete the temporary directory using shutil, ignore errors if there are any
        # read-only files

        shutil.rmtree(Path("test_grouped_fuzzy_sets"), ignore_errors=True)
