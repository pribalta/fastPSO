import unittest
from unittest.mock import patch

import numpy as np

from fastPSO.pso import PsoParameters, Particle, Bounds, Swarm
from fastPSO.pso import Pso, ObjectiveFunctionBase, Executor


class DummyObjectiveFunction(ObjectiveFunctionBase):
    def __init__(self):
        self._ret = 0.0

    def __call__(self, particle: Particle, *args, **kwargs):
        self._ret += 1.0
        return self._ret


class TestBounds(unittest.TestCase):

    def test_that_bounds_must_have_the_same_shape(self):
        with self.assertRaises(ValueError):
            Bounds(np.array([[1, 1], [1, 1]]), np.array([10, 10, 10]))

    def test_that_bounds_must_have_the_same_length(self):
        with self.assertRaises(ValueError):
            Bounds(np.array([[1, 1]]), np.array([10, 10, 10]))

    def test_that_bounds_must_be_1d(self):
        with self.assertRaises(ValueError):
            Bounds(np.array([[1, 1], [1, 1]]), np.array([10, 10, 10]))

    def test_that_bounds_must_be_1d_2(self):
        with self.assertRaises(ValueError):
            Bounds(np.array([[1, 1]]), np.array([[10, 10, 10], [1]]))

    def test_that_upper_bound_must_be_greater_equal(self):
        b = Bounds(np.array([1, 1]), np.array([10, 10]))

        self.assertTrue(np.all(np.equal(b.lower(), np.array([1, 1]))))
        self.assertTrue(np.all(np.equal(b.upper(), np.array([10, 10]))))

    def test_that_upper_bound_must_be_greater_equal_2(self):
        b = Bounds(np.array([1, 1]), np.array([1, 10]))

        self.assertTrue(np.all(np.equal(b.lower(), np.array([1, 1]))))
        self.assertTrue(np.all(np.equal(b.upper(), np.array([1, 10]))))

    def test_that_upper_bound_must_be_greater_equal_3(self):
        b = Bounds(np.array([1, 1]), np.array([1, 1]))

        self.assertTrue(np.all(np.equal(b.lower(), np.array([1, 1]))))
        self.assertTrue(np.all(np.equal(b.upper(), np.array([1, 1]))))

    def test_that_upper_bound_must_be_greater_equal_4(self):
        with self.assertRaises(ValueError):
            Bounds(np.array([1, 2]), np.array([1, 1]))

    def test_that_bouds_are_properly_stored(self):
        b = Bounds(np.array([1, 1]), np.array([10, 10]))

        self.assertTrue(np.all(np.equal(b.lower(), np.array([1, 1]))))
        self.assertTrue(np.all(np.equal(b.upper(), np.array([10, 10]))))


class TestPsoParameters(unittest.TestCase):

    def test_that_omega_must_be_between_0_and_1(self):
        with self.assertRaises(ValueError):
            PsoParameters(omega=-0.1, phip=0.5, phig=0.5)

    def test_that_omega_must_be_between_0_and_1_2(self):
        with self.assertRaises(ValueError):
            PsoParameters(omega=1.1, phip=0.5, phig=0.5)

    def test_that_omega_must_be_between_0_and_1_3(self):
        p = PsoParameters(omega=0.5, phip=0.5, phig=0.5)

        self.assertEqual(p.omega(), 0.5)
        self.assertEqual(p.phip(), 0.5)
        self.assertEqual(p.phig(), 0.5)

    def test_that_omega_must_be_between_0_and_1_4(self):
        p = PsoParameters(omega=0, phip=0.5, phig=0.5)

        self.assertEqual(p.omega(), 0)
        self.assertEqual(p.phip(), 0.5)
        self.assertEqual(p.phig(), 0.5)

    def test_that_omega_must_be_between_0_and_1_5(self):
        p = PsoParameters(omega=1, phip=0.5, phig=0.5)

        self.assertEqual(p.omega(), 1)
        self.assertEqual(p.phip(), 0.5)
        self.assertEqual(p.phig(), 0.5)

    def test_that_phip_must_be_between_0_and_1(self):
        with self.assertRaises(ValueError):
            PsoParameters(omega=0.5, phip=-0.1, phig=0.5)

    def test_that_phip_must_be_between_0_and_1_2(self):
        with self.assertRaises(ValueError):
            PsoParameters(omega=0.5, phip=1.1, phig=0.5)

    def test_that_phip_must_be_between_0_and_1_3(self):
        p = PsoParameters(omega=0.5, phip=0.5, phig=0.5)

        self.assertEqual(p.omega(), 0.5)
        self.assertEqual(p.phip(), 0.5)
        self.assertEqual(p.phig(), 0.5)

    def test_that_phip_must_be_between_0_and_1_4(self):
        p = PsoParameters(omega=0.5, phip=0, phig=0.5)

        self.assertEqual(p.omega(), 0.5)
        self.assertEqual(p.phip(), 0)
        self.assertEqual(p.phig(), 0.5)

    def test_that_phip_must_be_between_0_and_1_5(self):
        p = PsoParameters(omega=0.5, phip=1, phig=0.5)

        self.assertEqual(p.omega(), 0.5)
        self.assertEqual(p.phip(), 1)
        self.assertEqual(p.phig(), 0.5)

    def test_that_phig_must_be_between_0_and_1(self):
        with self.assertRaises(ValueError):
            PsoParameters(omega=0.5, phip=-0.1, phig=-0.1)

    def test_that_phig_must_be_between_0_and_1_2(self):
        with self.assertRaises(ValueError):
            PsoParameters(omega=0.5, phip=0.5, phig=1.1)

    def test_that_phig_must_be_between_0_and_1_3(self):
        p = PsoParameters(omega=0.5, phip=0.5, phig=0.5)

        self.assertEqual(p.omega(), 0.5)
        self.assertEqual(p.phip(), 0.5)
        self.assertEqual(p.phig(), 0.5)

    def test_that_phig_must_be_between_0_and_1_4(self):
        p = PsoParameters(omega=0.5, phip=0.5, phig=0)

        self.assertEqual(p.omega(), 0.5)
        self.assertEqual(p.phip(), 0.5)
        self.assertEqual(p.phig(), 0)

    def test_that_phig_must_be_between_0_and_1_5(self):
        p = PsoParameters(omega=0.5, phip=0.5, phig=1)

        self.assertEqual(p.omega(), 0.5)
        self.assertEqual(p.phip(), 0.5)
        self.assertEqual(p.phig(), 1)


class TestParticle(unittest.TestCase):

    @patch.object(Particle, '_calculate_initial_position')
    def test_that_particle_is_initialized_with_position(self, mock):
        mock.return_value = np.array([666])
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.5, 0.5, 0.5))

        self.assertTrue(mock.called)
        self.assertEqual(p.position(), np.array([666]))
        self.assertEqual(len(p._position), 1)

    @patch.object(Particle, '_calculate_initial_velocity')
    def test_that_particle_is_initialized_with_velocity(self, mock):
        mock.return_value = np.array([666])
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.5, 0.5, 0.5))

        self.assertTrue(mock.called)
        self.assertEqual(p.velocity(), np.array([666]))
        self.assertEqual(len(p._velocity), 1)

    def test_that_particle_is_initialized_without_score(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.5, 0.5, 0.5))

        self.assertEqual(len(p._score), 0)

    def test_that_bounds_are_correctly_assigned(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.5, 0.5, 0.5))

        self.assertEqual(p._bounds.lower(), np.array([1]))
        self.assertEqual(p._bounds.upper(), np.array([2]))

    def test_that_parameters_are_correctly_assigned(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        self.assertEqual(p._parameters.omega(), 0.15)
        self.assertEqual(p._parameters.phip(), 0.15)
        self.assertEqual(p._parameters.phig(), 0.15)

    def test_that_position_returns_the_current_position(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._position = list(reversed(range(10)))

        self.assertEqual(p.position(), p._position[-1])
        self.assertEqual(p.position(), 0)
        self.assertEqual(len(p._position), 10)

    def test_that_position_returns_the_current_position_2(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._position = list(range(10))

        self.assertEqual(p.position(), p._position[-1])
        self.assertEqual(p.position(), 9)
        self.assertEqual(len(p._position), 10)

    def test_that_velocity_returns_the_current_velocity(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._velocity = list(range(10))

        self.assertEqual(p.velocity(), p._velocity[-1])
        self.assertEqual(p.velocity(), 9)
        self.assertEqual(len(p._velocity), 10)

    def test_that_velocity_returns_the_current_velocity_2(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._velocity = list(range(10))

        self.assertEqual(p.velocity(), p._velocity[-1])
        self.assertEqual(p.velocity(), 9)
        self.assertEqual(len(p._velocity), 10)

    def test_that_best_position_returns_the_position_of_the_best_score(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._score = list(range(10))
        p._position = list(reversed(range(10)))

        self.assertEqual(len(p._score), 10)
        self.assertEqual(len(p._position), 10)
        self.assertEqual(p.best_position(), 0)

    def test_that_best_position_can_be_calculated_if_same_number_of_scores(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._score = list(range(9))
        p._position = list(reversed(range(10)))

        with self.assertRaises(ValueError):
            p.best_position()

    def test_that_best_score_can_be_calculated_if_scores_not_empty(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        with self.assertRaises(ValueError):
            p.best_score()

    def test_that_best_score_returns_the_highest_score(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._score = list(range(10))
        p._position = list(reversed(range(10)))

        self.assertEqual(len(p._score), 10)
        self.assertEqual(len(p._position), 10)
        self.assertEqual(p.best_score(), 9)

    def test_that_update_score_increases_the_amount_of_stored_scores(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        self.assertEqual(len(p._score), 0)

        p.update_score(666)

        self.assertEqual(len(p._score), 1)
        self.assertEqual(p.best_score(), 666)

    def test_that_last_movement_returns_distance(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._position = [np.array([1]), np.array([7])]

        self.assertEqual(p.last_movement(), 6)

    def test_that_last_movement_returns_distance_2d(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._position = [np.array([0, 0]), np.array([6, 0])]

        self.assertEqual(p.last_movement(), 6)

    def test_that_lastimprovement_returns_delta(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._score = [75, 0, 1000]

        self.assertEqual(p.last_improvement(), 1000)

    def test_that_lastimprovement_returns_delta_2(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._score = [75, 1000, 0]

        self.assertEqual(p.last_improvement(), -1000)

    def test_that_lastimprovement_returns_infinity_if_only_one_score(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._score = [0]

        self.assertEqual(p.last_improvement(), float("inf"))

    def test_that_lastimprovement_is_inf_if_not_enough_scores(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        self.assertEqual(p.last_improvement(), float('inf'))

    def test_that_lastimprovement_is_inf_if_not_enough_scores2(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._score = [0]

        self.assertEqual(p.last_improvement(), float('inf'))

    def test_that_lastimprovement_is_calculated(self):
        p = Particle(Bounds(np.array([1]), np.array([2])), PsoParameters(0.15, 0.15, 0.15))

        p._score = [0, 5]

        self.assertEqual(p.last_improvement(), 5)


class TestSwarm(unittest.TestCase):

    def test_that_swarm_has_to_be_created_with_positive_number_of_particles(self):
        with self.assertRaises(ValueError):
            Swarm(swarm_size=-1,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

    def test_that_swarm_has_to_be_created_with_positive_number_of_particles_2(self):
        with self.assertRaises(ValueError):
            Swarm(swarm_size=0,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

    def test_that_swarm_is_iterable(self):
        s = Swarm(swarm_size=10,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        particles = [p for p in s]

        self.assertEqual(len(particles), 10)

    def test_that_swarm_has_length(self):
        s = Swarm(swarm_size=10,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        particles = [p for p in s]

        self.assertEqual(len(particles), 10)
        self.assertEqual(len(s), 10)
        self.assertEqual(len(s), len(particles))

    @patch.object(Particle, 'update_score')
    def test_that_swarm_can_update_the_scores_of_all_particles(self, mock):
        s = Swarm(swarm_size=10,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        s.update_scores(list(range(10)))

        self.assertTrue(mock.called)
        self.assertEqual(mock.call_count, 10)

    def test_that_swarm_can_update_the_scores_of_all_particles_if_scores_equal_swarmsize(self):
        s = Swarm(swarm_size=10,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        with self.assertRaises(ValueError):
            s.update_scores(list(range(9)))

    @patch.object(Particle, 'last_improvement')
    def test_improvement(self, mock):
        s = Swarm(swarm_size=10,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        mock.return_value = 1

        self.assertTrue(s.still_improving())

    @patch.object(Particle, 'last_improvement')
    def test_improvement_2(self, mock):
        s = Swarm(swarm_size=10,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        mock.return_value = 0

        self.assertFalse(s.still_improving())

    @patch.object(Particle, 'last_movement')
    def test_movement(self, mock):
        s = Swarm(swarm_size=10,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        mock.return_value = 1

        self.assertTrue(s.still_moving())

    @patch.object(Particle, 'last_movement')
    def test_movement_2(self, mock):
        s = Swarm(swarm_size=10,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        mock.return_value = 0

        self.assertFalse(s.still_moving())

    @patch.object(Swarm, 'best_position')
    @patch.object(Particle, 'update')
    def test_that_update_velocity_is_appleid_to_all_particles(self, mock_update, mock_position):
        s = Swarm(swarm_size=10,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        mock_position.return_value = 666

        s.update_velocity()

        self.assertTrue(mock_update.called)
        self.assertEqual(mock_update.call_count, 10)

    @patch.object(Particle, 'best_position')
    @patch.object(Particle, 'best_score')
    def test_best_position(self, mock_score, mock_position):
        s = Swarm(swarm_size=10,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        mock_position.side_effect = list(range(10))
        mock_score.side_effect = list(reversed(range(10)))

        self.assertEqual(s.best_position(), 0)

    @patch.object(Particle, 'best_score')
    def test_best_score(self, mock_score):
        s = Swarm(swarm_size=10,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        mock_score.side_effect = list(reversed(range(10)))

        self.assertEqual(s.best_score(), 9)


class TestExecutor(unittest.TestCase):

    def test_that_thread_count_must_be_positive(self):
        with self.assertRaises(ValueError):
            Executor(DummyObjectiveFunction(), threads=0)

    def test_that_thread_count_must_be_positive_2(self):
        with self.assertRaises(ValueError):
            Executor(DummyObjectiveFunction(), threads=-1)

    def test_that_thread_count_must_be_positive_3(self):
        ex = Executor(DummyObjectiveFunction(), threads=10)

        self.assertEqual(ex._threads, 10)

    @patch.object(Swarm, '__len__')
    @patch.object(Swarm, 'update_scores')
    def test_that_maximum_number_of_threads_is_limited_by_swarm_size(self, mock_update, mock_size):
        s = Swarm(swarm_size=1,
                  bounds=Bounds(np.array([1]), np.array([2])),
                  parameters=PsoParameters(0.15, 0.15, 0.15),
                  minimum_improvement=10e-8,
                  minimum_step=10e-8)

        ex = Executor(DummyObjectiveFunction(), threads=10)

        mock_size.return_value = 1

        ex.calculate_scores(s)

        self.assertTrue(mock_size.called)
        self.assertEqual(mock_update.call_count, 1)


class TestPso(unittest.TestCase):

    def test_that_maximum_number_of_iterations_is_positive(self):
        with self.assertRaises(ValueError):
            Pso(DummyObjectiveFunction(), np.array([0]), np.array([10]), 10, 0.5, 0.5, 0.5, -1)

    def test_that_maximum_number_of_iterations_is_positive_2(self):
        with self.assertRaises(ValueError):
            Pso(DummyObjectiveFunction(), np.array([0]), np.array([10]), 10, 0.5, 0.5, 0.5, 0)
