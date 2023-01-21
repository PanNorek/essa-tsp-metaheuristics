from src.algos import PFSPAlgorithm
import random
from typing import Union, Any
import pandas as pd
from src.utils import solve_it, Result, NEHInsertion


class NEH(PFSPAlgorithm):
    NAME = "NEH"

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(verbose)
        self._neigh = NEHInsertion(path_length=None)

    @solve_it
    def _solve(
        self,
        distances: pd.DataFrame,
        random_seed: Union[int, None] = None,
        start_order: Union[int, None] = None,
    ) -> int:
        # specific implementation for NEH algorithm
        # jobs ordered incresingly by cummulative time on all machines
        order = distances.sum(axis=1).sort_values().index.to_list()

        # the first element is job chosen as starting point
        self._path = [order[0]]

        # add jobs in sequence from less to most time consuming
        # insert new job into the best place (that minimize cost function)
        for x in range(1, len(distances)):
            self._path.append(order[x])
            self._path = self._neigh.switch(path=self._path, distances=distances)

        # calculates the final cost function
        distance = self._get_order_time(path=self._path, distances=distances)
        # returns Result object
        return Result(algorithm=self, path=self._path, distance=distance)
