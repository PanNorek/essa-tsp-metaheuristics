from typing import Union, List
import pandas as pd
from ..utils import StopAlgorithm
from .swapping_algorithm import SwappingAlgorithm

class HillClimber(SwappingAlgorithm):
    """Hill Climber Algorithm

    Parameters
    ----------
    neigh_type : str, optional
        Type of neighborhood, by default None
    n_iter : int, optional
        Number of iterations, by default 30
    verbose : bool, optional
        Print progress, by default False
    """

    NAME = "HILL CLIMBER"

    def _iterate_steps(
        self, distances: pd.DataFrame, swaps: List[tuple]
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
        best_swap, best_distance = self._find_best_swap(
            swaps=swaps, distances=distances
        )
        # distance gain
        gain = self.history[-1] - best_distance
        # break condition
        if gain <= 0:
            raise StopAlgorithm(iteration=self._i, distance=best_distance)
        if self._verbose:
            print(f"best swap: {best_swap} - gain: {gain}")
            print(f"step {self._i}: distance: {best_distance}")

        # new path that shortens the distance
        self._path = self._swap_elements(swap=best_swap)
        # adding new best distance to distances history
        self.history.append(best_distance)
        return best_distance, best_swap
