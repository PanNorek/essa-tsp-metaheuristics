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

    def __init__(self,
                 tabu_length: int = 3,
                 neigh_type: str = "swap",
                 n_iter: int = 100,
                 verbose: bool = False,
                 ) -> None:
        super().__init__(
            neigh_type=neigh_type,
            n_iter=n_iter,
            verbose=verbose
        )
        self._tabu_list = Queue(length=tabu_length)

    def _iterate_steps(self, distances: pd.DataFrame) -> None:
        # start new iteration
        self._i += 1
        new_path = self._switch(distances=distances,
                                how='best',
                                exclude=self._tabu_list)
        new_distance = self._get_path_distance(path=new_path,
                                               distances=distances)
        # new path that minimizes distance
        self._path = new_path
        # removing first swap from tabu list only if list is full
        self._tabu_list.dequeue()
        # adding new swap to tabu list
        self._tabu_list.enqueue(self._last_switch)
        if self._verbose:
            gain = self.history[-1] - new_distance
            print(f"best switch: {self._last_switch_comment} - gain: {gain}")
            print(f"step {self._i}: distance: {new_distance}")
            print(f"tabu list: {self._tabu_list}")

        # adding new best distance to distances history
        self.history.append(new_distance)
