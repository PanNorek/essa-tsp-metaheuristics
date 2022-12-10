from .swapping_algorithm import SwappingAlgorithm
from ..utils import time_it
from typing import Tuple, Union, List
import pandas as pd
from multiprocessing import Pool
from ..utils import StopAlgorithm


def unwrap_self_f(arg, **kwarg):
    return HillClimber.solve_multistart_parallel(*arg, **kwarg)


class HillClimber(SwappingAlgorithm):
    """ Hill Climber Algorithm """
    NAME = 'HILL CLIMBER'

    def _iterate_steps(self,
                       distances: pd.DataFrame,
                       swaps: List[tuple]
                       ) -> Union[int, None]:
        best_distance, best_swap = super()._iterate_steps(distances=distances,
                                                          swaps=swaps)
        # distance gain
        gain = self.history[-1] - best_distance
        # break condition
        if gain <= 0:
            raise StopAlgorithm(iter=self._i, distance=best_distance)
        if self._verbose:
            print(f'best swap: {best_swap} - gain: {gain}')
            print(f'step {self._i}: distance: {best_distance}')

        # adding new best distance to distances history
        self.history.append(best_distance)
        return best_distance, best_swap

    def solve_multistart(self,
                         distance_matrix: pd.DataFrame,
                         num_iter: int = 20,
                         num_starts: int = 10
                         ) -> Tuple[int, str]:
        """Solve TSP problem with Hill Climber Algorithm with multiple starts

        Args:
            distance_matrix (pd.DataFrame): Distance matrix
            num_iter (int): Number of iterations
            num_starts (int): Number of starts

        Returns:
            distance (int): Total distance
            path (str): Salesman path
        """

        # TODO: add multiprocessing
        results = []
        for i in range(num_starts):
            results.append(self.solve(distance_matrix,
                                      num_iter=num_iter,
                                      return_tuple=True))
            print(f"Start {i}")

        return min(results, key=lambda x: x[0])
        # return results

    def solve_multistart_parallel(self,
                                  distance_matrix: pd.DataFrame,
                                  num_iter: int = 20,
                                  num_starts: int = 0
                                  ) -> Tuple[int, str]:
        """Solve TSP problem with Hill Climber Algorithm with multiple starts

        Args:
            distance_matrix (pd.DataFrame): Distance matrix
            num_iter (int): Number of iterations
            num_starts (int): Number of starts

        Returns:
            distance (int): Total distance
            path (str): Salesman path
        """
        pool = Pool(processes=2)
        pool.map(unwrap_self_f, [(self, distance_matrix, num_iter, num_starts)])
