from typing import List
from multiprocessing import Pool

import numpy as np


class ObjectiveFunctionBase(object):
    def __call__(self, *args, **kwargs):
        pass


class PsoParameters(object):
    def __init__(self,
                 omega: float = 0.5,
                 phip: float = 0.5,
                 phig: float = 0.5):
        self._omega = omega
        self._phip = phip
        self._phig = phig

    def omega(self):
        return self._omega

    def phip(self):
        return self._phip

    def phig(self):
        return self._phig


class Particle(object):
    def __init__(self,
                 lower_bound: np.ndarray,
                 upper_bound: np.ndarray,
                 parameters: PsoParameters):
        assert lower_bound.dtype == upper_bound.dtype, \
            "Upper and lower bound must share the same type. Found {} and {}".format(lower_bound.type,
                                                                                     upper_bound.type)
        assert np.greater_equal(upper_bound, lower_bound), \
            "Upper bound values must be greater or equal than lower bound values. Received {} and {}".format(
                upper_bound,
                lower_bound)

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._parameters = parameters

        self._position = [self._initialize_position()]
        self._velocity = [self._initialize_velocity()]
        self._score = []

    def position(self) -> np.ndarray:
        return self._position[-1]

    def velocity(self) -> np.ndarray:
        return self._velocity[-1]

    def best_position(self) -> np.ndarray:
        return self._position[np.argsort(self._score)[-1]]

    def best_score(self) -> np.ndarray:
        return self._score[np.argsort(self._score)[-1]]

    def update_velocity(self,
                        swarm_best: np.ndarray):
        assert self._score, "Cannot update velocity while scores are empty. Evaluate first."

        rp, rg = self._initialize_random_coefficients()

        self._velocity.append(self._parameters.omega() * self.velocity()
                              + self._parameters.phip() * rp * (self.best_position() - self.position())
                              + self._parameters.phig() * rg * (swarm_best - self.position()))
        self._position.append(self._update_position())

    def update_score(self, score: float) -> None:
        self._score.append(score)

    def _initialize_position(self) -> np.ndarray:
        return np.array([np.random.uniform(low, high)
                         for low, high in zip(self._lower_bound,
                                              self._upper_bound)]).astype(self._lower_bound.dtype)

    def _initialize_velocity(self) -> np.ndarray:
        return np.array([np.random.uniform(-(high - low), high - low)
                         for low, high in zip(self._lower_bound,
                                              self._upper_bound)]).astype(self._lower_bound.dtype)

    def _initialize_random_coefficients(self):
        return np.random.uniform(0, 1), np.random.uniform(0, 1)

    def _update_position(self):
        new_position = self._position[-1] + self._velocity[-1]

        for i in range(new_position.size):
            if self._lower_bound[i] > new_position[i]:
                new_position[i] = self._lower_bound[i]
            elif self._upper_bound < new_position[i]:
                new_position[i] = self._upper_bound[i]

        return new_position


class Executor(object):
    def __init__(self,
                 objective_function: ObjectiveFunctionBase,
                 threads: int,
                 minimum_improvement: float,
                 minimum_step: float):
        assert threads > 0, "Number of threads must be greater than zero"
        self._objective_function = objective_function
        self._threads = threads
        self._minimum_improvement = minimum_improvement
        self._minimum_step = minimum_step

    def calculate_scores(self,
                         particles: List[Particle]):
        with Pool(min(self._threads, len(particles))) as pool:
            return pool.starmap(self._objective_function, zip(particles), chunksize=1)

    def still_improving(self) -> bool:
        return True

    def still_moving(self) -> bool:
        return True


class Pso(object):
    def __init__(self,
                 objective_function: ObjectiveFunctionBase,
                 lower_bound: np.ndarray,
                 upper_bound: np.ndarray,
                 swarm_size: int,
                 omega: float = 0.5,
                 phip: float = 0.5,
                 phig: float = 0.5,
                 maximum_iterations: int = 100,
                 minimum_step: float = 10e-8,
                 minimum_improvement: float = 10e-8,
                 threads: int = 1,
                 verbose: bool = False):

        assert lower_bound.shape == upper_bound.shape, \
            "Lower and upper bound must have the same shape. Received {} and {}".format(lower_bound.shape,
                                                                                        upper_bound.shape)

        assert swarm_size > 0, "Swarm size must be greater than zero"
        assert 1 >= omega >= 0, "Value for omega should be [0.0, 1.0]"
        assert 1 >= phip >= 0, "Value for phip should be [0.0, 1.0]"
        assert 1 >= phig >= 0, "Value for phig should be [0.0, 1.0]"
        assert maximum_iterations > 0, "Maximum number of iterations must be greater than zero"

        self._parameters = PsoParameters(omega, phip, phig)
        self._particles = [Particle(lower_bound, upper_bound, self._parameters) for _ in range(swarm_size)]
        self._maximum_iterations = maximum_iterations

        self._executor = Executor(objective_function,
                                  threads,
                                  minimum_improvement,
                                  minimum_step)

    def run(self):
        for i in range(self._maximum_iterations):
            self._executor.calculate_scores(self._particles)

            if not self._executor.still_improving() \
                    or not self._executor.still_moving():
                break


def main():
    pso = Pso(None, np.array([1, 1, 1]), np.array([10, 10, 10]), 1)
    pass


if __name__ == '__main__':
    main()
