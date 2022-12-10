from .swapping_algorithm import SwappingAlgorithm
from ..utils import time_it, Result
from typing import Tuple, Union, List
import pandas as pd
from multiprocessing import Pool
from ..utils import StopAlgorithm


class HillClimber(SwappingAlgorithm):
    """ Hill Climber Algorithm """
    NAME = 'HILL CLIMBER'

    def _iterate_steps(self,
                       distances: pd.DataFrame,
                       swaps: List[tuple]
                       ) -> Union[int, None]:
        """Iterate steps in Hill Climber Algorithm

        Args:
            distances (pd.DataFrame): Distance matrix
            swaps (List[tuple]): List of possible swaps

        Returns:
            distance (int): Total distance
            swap (tuple): Swap
        """

        # start new iteration
        self._i += 1
        best_swap, best_distance = self._find_best_swap(swaps=swaps,
                                                        distances=distances)
        # distance gain
        gain = self.history[-1] - best_distance
        # break condition
        if gain <= 0:
            raise StopAlgorithm(iter=self._i, distance=best_distance)
        if self._verbose:
            print(f'best swap: {best_swap} - gain: {gain}')
            print(f'step {self._i}: distance: {best_distance}')

        # new path that shortens the distance
        self._path = self._swap_elements(swap=best_swap)
        # adding new best distance to distances history
        self.history.append(best_distance)
        return best_distance, best_swap


# class HillClimberMultistart(HillClimber):
#     """ Hill Climber Algorithm with multiple starts """
#     NAME = 'HILL CLIMBER MULTISTART'

#     @time_it
#     def solve_multistart(self,
#               distances: pd.DataFrame,
#               num_starts: int = 10,
#               **kwargs
#               ) -> Result:
#         """Solve TSP problem with Hill Climber Algorithm with multiple starts

#         Args:
#             distance_matrix (pd.DataFrame): Distance matrix
#             num_iter (int): Number of iterations
#             num_starts (int): Number of starts

#         Returns:
#             Result: Result object
#         """
#         results = []
#         history = []

#         # both methods are equivalent in time, the second one is more readable
#         # tic = time.time()
#         # list(map(lambda x: self.solve(distances, **kwargs), range(num_starts)))
#         # toc = time.time()

#         for i in range(1, num_starts + 1):
#             results.append(self.solve(distances,
#                                       **kwargs
#                                       ))
#             if len(history) == 0:
#                 history.append(results[-1].best_distance)
#             else:
#                 if results[-1].best_distance < history[-1]:
#                     history.append(results[-1].best_distance)
#                 # else:
#                 #     history.append(history[-1])
#             print(f'No. of start: {i}')

#         self.history = history
#         best_result = min(results, key=lambda x: x.best_distance)
#         best_result.no_starts = num_starts
#         return best_result
         