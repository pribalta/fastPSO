import numpy as np


class Particle(object):
    def __init__(self,
                 lower_bound: np.ndarray,
                 upper_bound: np.ndarray):
        assert lower_bound.dtype == upper_bound.dtype, \
            "Upper and lower bound must share the same type. Found {} and {}".format(lower_bound.type,
                                                                                     upper_bound.type)
        assert np.greater_equal(upper_bound, lower_bound), \
            "Upper bound values must be greater or equal than lower bound values. Received {} and {}".format(
                upper_bound,
                lower_bound)

        self._position = [self._initialize_position(lower_bound, upper_bound)]
        self._velocity = [self._initialize_velocity(lower_bound, upper_bound)]
        self._score = []

    def position(self) -> np.ndarray:
        return self._position[-1]

    def velocity(self) -> np.ndarray:
        return self._velocity[-1]

    def best_position(self) -> np.ndarray:
        return self._position[np.argmax(self._score)]

    def best_score(self) -> np.ndarray:
        return self._score[np.argmax(self._score)]

    def update_velocity(self,
                        omega: float,
                        phip: float,
                        phig: float,
                        swarm_best: np.ndarray):
        assert self._score, "Cannot update velocity while scores are empty. Evaluate first."

        rp, rg = self._initialize_random_coefficients()

        self._velocity.append(omega * self.velocity()
                              + phip * rp * (self.best_position() - self.position())
                              + phig * rg * (swarm_best - self.position()))

    def update_score(self, score: float) -> None:
        self._score.append(score)

    def _initialize_position(self,
                             lower_bound: np.ndarray,
                             upper_bound: np.ndarray) -> np.ndarray:
        return np.array([np.random.uniform(low, high)
                         for low, high in zip(lower_bound, upper_bound)]).astype(lower_bound.dtype)

    def _initialize_velocity(self,
                             lower_bound: np.ndarray,
                             upper_bound: np.ndarray) -> np.ndarray:
        return np.array([np.random.uniform(-(high - low), high - low)
                         for low, high in zip(lower_bound, upper_bound)]).astype(lower_bound.dtype)

    def _initialize_random_coefficients(self):
        return np.random.uniform(0, 1), np.random.uniform(0, 1)
