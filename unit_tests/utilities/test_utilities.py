import unittest

from soft.utilities.performance import is_debugger_active, performance_boost


class TestUtilities(unittest.TestCase):
    def test_performance_boost(self) -> None:
        """
        Test that a performance boost is enabled when the debugger is not active, or vice-versa.

        Returns:
            None
        """
        self.assertEqual(not performance_boost(), is_debugger_active())
