import numpy as np

from pyPso.swarm import Particle


class Pso(object):
    def __init__(self,
                 objective_function,
                 lower_bound: np.ndarray,
                 upper_bound: np.ndarray,
                 swarm_size: int,
                 omega: float = 0.5,
                 phip: float = 0.5,
                 phig: float = 0.5,
                 maximum_iterations: int = 100,
                 minimum_step: float = 10e-8,
                 minimum_improvement: float = 10e-8,
                 cpu_threads: int = 1,
                 gpu_threads: int = 1,
                 verbose: bool = False):

        assert lower_bound.shape == upper_bound.shape, \
            "Lower and upper bound must have the same shape. Received {} and {}".format(lower_bound.shape,
                                                                                        upper_bound.shape)

        self._particles = [Particle(lower_bound, upper_bound) for _ in range(swarm_size)]


def main():
    pso = Pso(None, np.array([1,1,1]), np.array([10,10,10]), 1)
    pass

if __name__ == '__main__':
    main()


