from typing import Union, List
import pandas as pd
from .swapping_algorithm import SwappingAlgorithm
from ..utils import Queue


class TabuSearch(SwappingAlgorithm):
    """Tabu Search Algorithm

    Parameters
    ----------
    tabu_length: int
        length of tabu list, default is 3
    neigh_type: str
        type of neighbourhood, default is None
    n_iter: int
        number of iterations, default is 100
    verbose: bool
        print progress, default is False
    """

    NAME = "TABU SEARCH"

    def __init__(
        self,
        tabu_length: int = 3,
        neigh_type: str = None,
        n_iter: int = 100,
        verbose: bool = False,
    ) -> None:
        super().__init__(neigh_type=neigh_type, n_iter=n_iter, verbose=verbose)
        self._tabu_list = Queue(length=tabu_length)

    def _iterate_steps(
        self, distances: pd.DataFrame, swaps: List[tuple]
    ) -> Union[int, None]:
        # start new iteration
        self._i += 1
        best_swap, best_distance = self._find_best_swap(
            swaps=swaps, distances=distances
        )
        # new path that minimizes distance
        self._path = self._swap_elements(swap=best_swap)
        # removing first swap from tabu list only if list is full
        self._tabu_list.dequeue()
        # adding new swap to tabu list
        self._tabu_list.enqueue(best_swap)
        gain = self.history[-1] - best_distance
        if self._verbose:
            print(f"best swap: {best_swap} - gain: {gain}")
            print(f"step {self._i}: distance: {best_distance}")
            print(f"tabu list: {self._tabu_list}")

        # adding new best distance to distances history
        self.history.append(best_distance)
        return best_distance, best_swap

    def _find_best_swap(self, swaps: List[tuple], distances: pd.DataFrame) -> pd.Series:
        distances_df = self._get_swaps_df(swaps=swaps, distances=distances)
        # dropping all swaps that are in tabu list
        distances_df = distances_df[~distances_df[self.SWAP].isin(self._tabu_list)]
        # taking row of the best swaps
        best_swap = distances_df.iloc[0]
        return best_swap
