import unittest

import numpy as np

from YACS.yacs import Config
from soft.utilities.reproducibility import load_configuration, parse_configuration


class TestReproducibility(unittest.TestCase):
    def _parse_configuration(self, config, before_parse, after_parse) -> None:
        """
        A helper to test that the parse configuration can be reversed.

        Args:
            config:
            before_parse:
            after_parse:

        Returns:

        """
        with config.unfreeze():
            config.fuzzy.t_norm.yager = before_parse

        self.assertEqual(config.fuzzy.t_norm.yager, before_parse)
        parsed_config = parse_configuration(config, reverse=True)
        self.assertEqual(parsed_config.fuzzy.t_norm.yager, after_parse)

    def test_parse_configuration(self) -> None:
        """
        Test that the parse configuration can be reversed.

        Returns:
            None
        """
        config: Config = load_configuration()
        self._parse_configuration(config, np.e, "euler")
        self._parse_configuration(config, 50.0, 50.0)
        self._parse_configuration(config, (1 + 5**0.5) / 2, "golden")
        self._parse_configuration(config, "algebraic_product", "algebraic_product")
