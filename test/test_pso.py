import unittest
from unittest.mock import patch

import numpy as np

from pyPso.pso import PsoParameters, Particle
from pyPso.pso import Pso, ObjectiveFunctionBase, Executor


class DummyObjectiveFunction(ObjectiveFunctionBase):
    def __init__(self):
        self._ret = 0.0

    def __call__(self, particle: Particle, *args, **kwargs):
        self._ret += 1.0
        return self._ret

class TestPso(unittest.TestCase):

    def test_that_assert_is_thrown_if_boundaries_are_not_the_same_length(self):
        with self.assertRaises(Exception):
            Pso(DummyObjectiveFunction(), np.array([1, 1]), np.array([10, 10, 10]), 1)

    def test_that_assert_is_thrown_if_max_iters_are_zero(self):
        with self.assertRaises(Exception):
            Pso(DummyObjectiveFunction(), np.array([1, 1]), np.array([10, 10]), 1, 0)

    def test_that_assert_is_thrown_if_max_iters_are_less_than_zero(self):
        with self.assertRaises(Exception):
            Pso(DummyObjectiveFunction(), np.array([1, 1]), np.array([10, 10]), 1, -10)

    def test_that_assert_is_thrown_if_swarm_size_is_zero(self):
        with self.assertRaises(Exception):
            Pso(DummyObjectiveFunction(), np.array([1, 1]), np.array([10, 10]), 0, 1)

    def test_that_assert_is_thrown_if_swarm_size_is_less_than_zero(self):
        with self.assertRaises(Exception):
            Pso(DummyObjectiveFunction(), np.array([1, 1]), np.array([10, 10]), -1, 1)

class TestExecutor(unittest.TestCase):

    def test_that_number_of_threads_cannot_be_zero(self):
        with self.assertRaises(Exception):
            Executor(DummyObjectiveFunction(), 0, 0, 0)

    def test_that_number_of_threads_is_greater_than_zero(self):
        with self.assertRaises(Exception):
            Executor(DummyObjectiveFunction(), -1, 0, 0)

    def test_that_objective_function_is_called(self):
        ex = Executor(DummyObjectiveFunction(), 1, 0, 0)
        score = ex.calculate_scores([Particle(np.array([2.0]), np.array([5.0]), PsoParameters())])

        self.assertEqual(len(score), 1)
        self.assertEqual(score[0], 1.0)

    def test_that_executor_returns_various_results(self):
        ex = Executor(DummyObjectiveFunction(), 1, 0, 0)
        score = ex.calculate_scores([Particle(np.array([2.0]), np.array([5.0]), PsoParameters()),
                                     Particle(np.array([2.0]), np.array([5.0]), PsoParameters())])

        self.assertEqual(len(score), 2)
        self.assertEqual(score[0], 1.0)
        self.assertEqual(score[1], 1.0)

class TestParticle(unittest.TestCase):

    def test_that_assert_is_thrown_if_boundaries_are_not_the_same_type(self):
        with self.assertRaises(Exception):
            Particle(np.array([1]), np.array([2.0]), PsoParameters())

    def test_that_upper_bound_must_be_greater_or_equal_than_lower_bound(self):
        Particle(np.array([2]), np.array([2]), PsoParameters())
        Particle(np.array([2]), np.array([3]), PsoParameters())

        with self.assertRaises(Exception):
            Particle(np.array([3]), np.array([2]), PsoParameters())

    def test_that_position_is_float_when_bounds_are_float(self):
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        self.assertEqual(particle.position().dtype, np.array([2.0]).dtype)

    def test_that_position_is_int_when_bounds_are_int(self):
        particle = Particle(np.array([2]), np.array([5]), PsoParameters())

        self.assertEqual(particle.position().dtype, np.array([2]).dtype)

    def test_that_velocity_is_float(self):
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        self.assertEqual(particle.velocity().dtype, np.array([2.0]).dtype)

    @patch.object(Particle, '_initialize_position')
    @patch.object(Particle, '_initialize_velocity')
    def test_that_position_and_velocity_are_initialized_when_creating_a_particle(self, _initialize_velocity, _initialize_position):
        Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        self.assertTrue(_initialize_position.called)
        self.assertTrue(_initialize_position.called)

    def test_that_position_and_velocity_are_of_length_one_on_creation(self):
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        self.assertTrue(len(particle._position), 1)
        self.assertTrue(len(particle._velocity), 1)

    def test_that_score_is_empty_on_creation(self):
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        self.assertEqual(len(particle._score), 0)

    def test_that_velocity_cannot_be_updated_before_calculating_fitness(self):
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        with self.assertRaises(Exception):
            particle.update_velocity(None)

    def test_that_velocity_can_be_updated_after_calculating_fitness(self):
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        particle.update_score(1.0)

        particle.update_velocity(np.array([3.0]))

    @patch.object(Particle, '_update_position')
    def test_that_updating_velocity_updates_position(self, mock):
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        particle.update_score(1.0)

        particle.update_velocity(np.array([3.0]))

        self.assertTrue(mock.called)

    @patch.object(Particle, '_initialize_random_coefficients')
    def test_that_random_coefficients_are_calculated_when_updating_velocity(self, mock):
        mock.return_value = (0.5, 0.5)
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        particle.update_score(1.0)

        particle.update_velocity(np.array([3.0]))

        self.assertTrue(mock.called)

    def test_position_returns_the_latest_position(self):
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        particle._position = range(0, 10)
        self.assertEqual(particle.position(), 9)

    def test_velocity_returns_the_latest_position(self):
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        particle._velocity = range(0, 10)
        self.assertEqual(particle.velocity(), 9)

    def test_best_position_returns_the_best_position(self):
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        particle._position = range(10, 20)
        particle._score = range(0, 10)
        self.assertEqual(particle.best_position(), 19)

    def test_best_score_returns_the_best_score(self):
        particle = Particle(np.array([2.0]), np.array([5.0]), PsoParameters())

        particle._score = range(0, 10)
        self.assertEqual(particle.best_score(), 9)


