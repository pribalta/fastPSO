import unittest

import numpy as np

from pyPso.pso import Pso


class TestPso(unittest.TestCase):

    def test_that_assert_is_thrown_if_boundaries_are_not_the_same_length(self):
        with self.assertRaises(Exception):
            Pso(None, np.array([1, 1]), np.array([10, 10, 10]), 1)

