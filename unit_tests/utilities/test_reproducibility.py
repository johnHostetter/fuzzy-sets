import unittest

from YACS.yacs import Config
from soft.utilities.reproducibility import load_configuration


class TestReproducibility(unittest.TestCase):
    def test_load_configuration(self) -> None:
        """
        Test that the configuration is loaded correctly (e.g., the learning rate is a float).

        Returns:
            None
        """
        config: Config = load_configuration()
        self.assertTrue(isinstance(config.training.learning_rate, float))
