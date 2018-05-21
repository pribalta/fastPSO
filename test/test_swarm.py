import unittest
from unittest.mock import patch

import numpy as np

from pyPso.swarm import Particle


class TestParticle(unittest.TestCase):

    def test_that_assert_is_thrown_if_boundaries_are_not_the_same_type(self):
        with self.assertRaises(Exception):
            Particle(np.array([1]), np.array([2.0]))

    def test_that_upper_bound_must_be_greater_or_equal_than_lower_bound(self):
        Particle(np.array([2]), np.array([2]))
        Particle(np.array([2]), np.array([3]))

        with self.assertRaises(Exception):
            Particle(np.array([3]), np.array([2]))

    def test_that_position_is_float_when_bounds_are_float(self):
        particle = Particle(np.array([2.0]), np.array([5.0]))

        self.assertEqual(particle.position().dtype, np.array([2.0]).dtype)

    def test_that_position_is_int_when_bounds_are_int(self):
        particle = Particle(np.array([2]), np.array([5]))

        self.assertEqual(particle.position().dtype, np.array([2]).dtype)

    def test_that_velocity_is_float(self):
        particle = Particle(np.array([2.0]), np.array([5.0]))

        self.assertEqual(particle.velocity().dtype, np.array([2.0]).dtype)

    @patch.object(Particle, '_initialize_position')
    @patch.object(Particle, '_initialize_velocity')
    def test_that_position_and_velocity_are_initialized_when_creating_a_particle(self, _initialize_velocity, _initialize_position):
        Particle(np.array([2.0]), np.array([5.0]))

        self.assertTrue(_initialize_position.called)
        self.assertTrue(_initialize_position.called)

    def test_that_position_and_velocity_are_of_length_one_on_creation(self):
        particle = Particle(np.array([2.0]), np.array([5.0]))

        self.assertTrue(len(particle._position), 1)
        self.assertTrue(len(particle._velocity), 1)

    def test_that_score_is_empty_on_creation(self):
        particle = Particle(np.array([2.0]), np.array([5.0]))

        self.assertEqual(len(particle._score), 0)

    def test_that_velocity_cannot_be_updated_before_calculating_fitness(self):
        particle = Particle(np.array([2.0]), np.array([5.0]))

        with self.assertRaises(Exception):
            particle.update_velocity(0,0,0,None)

    def test_that_velocity_can_be_updated_after_calculating_fitness(self):
        particle = Particle(np.array([2.0]), np.array([5.0]))

        particle.update_score(1.0)

        particle.update_velocity(0.5, 0.5 ,0.5, np.array([3.0]))

    @patch.object(Particle, '_initialize_random_coefficients')
    def test_that_random_coefficients_are_calculated_when_updating_velocity(self, mock):
        mock.return_value = (0.5, 0.5)
        particle = Particle(np.array([2.0]), np.array([5.0]))

        particle.update_score(1.0)

        particle.update_velocity(0.5, 0.5 ,0.5, np.array([3.0]))

        self.assertTrue(mock.called)

    def test_position_returns_the_latest_position(self):
        particle = Particle(np.array([2.0]), np.array([5.0]))

        particle._position = range(0, 10)
        self.assertTrue(particle.position(), 9)

    def test_velocity_returns_the_latest_position(self):
        particle = Particle(np.array([2.0]), np.array([5.0]))

        particle._velocity = range(0, 10)
        self.assertTrue(particle.velocity(), 9)

    def test_best_position_returns_the_best_position(self):
        particle = Particle(np.array([2.0]), np.array([5.0]))

        particle._position = range(10, 20)
        particle._score = range(0, 10)
        self.assertTrue(particle.best_position(), 19)

    def test_best_score_returns_the_best_score(self):
        particle = Particle(np.array([2.0]), np.array([5.0]))

        particle._score = range(0, 10)
        self.assertTrue(particle.best_score(), 9)

