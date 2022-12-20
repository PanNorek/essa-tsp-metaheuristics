from abc import abstractmethod
from typing import Union, List
import pandas as pd
from .algorithm import Algorithm
from ..utils import StopAlgorithm, Result, time_it, get_path_distance


class SwappingAlgorithm(Algorithm):
    """Swapping Algorithm"""

    DISTANCE = "distance"
    SWAP = "swap"

    def __init__(self,
                 neigh_type: str = "swap",
                 n_iter: int = 30,
                 verbose: bool = False,
                 inversion_window: Union[int, None] = None
                 ) -> None:
        super().__init__(neigh_type=neigh_type,
                         verbose=verbose,
                         inversion_window=inversion_window)
        self._n_iter = n_iter
        # current iteration
        self._i = 0

    @time_it
    def solve(self,
              distances: pd.DataFrame,
              random_seed: Union[int, None] = None,
              start_order: Union[list, None] = None
              ) -> Result:
        super().solve(distances=distances, random_seed=random_seed)
        self._path = start_order or self._get_random_path(indices=distances.index)
        # distance of the path at the beginning
        distance = get_path_distance(path=self._path, distances=distances)
        # list of distances at i iteration
        self.history = [distance]

        if self._verbose:
            print(f"--- {self.NAME} ---")
            print(f"step {self._i}: distance: {distance}")

        for _ in range(self._n_iter):
            try:
                self._iterate_steps(distances=distances)
            except StopAlgorithm as exc:
                if self._verbose:
                    print(exc.message)
                break
        # return result object
        return Result(
            algorithm=self,
            path=self._path,
            best_distance=min(self.history),
            distance_history=self.history,
        )

    @abstractmethod
    def _iterate_steps(self, distances: pd.DataFrame) -> None:
        pass

    def __str__(self) -> str:
        mes = super().__str__()
        mes += f"""\nn_iter: {self._n_iter}"""
        return mes
