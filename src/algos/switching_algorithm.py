from abc import abstractmethod
from typing import Union, Iterable
import pandas as pd
from .iterating_algorithm import IteratingAlgorithm
from ..utils import (
    StopAlgorithm,
    Result,
    solve_it,
    get_path_distance,
    Queue
)


class SwitchingAlgorithm(IteratingAlgorithm):
    """Swapping Algorithm"""

    DISTANCE = "distance"
    SWITCH = "switch"

    @solve_it
    def _solve(self,
               distances: pd.DataFrame,
               random_seed: Union[int, None] = None,
               start_order: Union[list, None] = None
               ) -> Result:
        self._path = self._setup_start(
            distances=distances,
            random_seed=random_seed,
            start_order=start_order
        )
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
            distance=min(self.history),
            distance_history=self.history,
        )

    def _switch(self,
                distances: pd.DataFrame,
                how: str = "best",
                exclude: Union[Queue, None] = None,
                ) -> list:
        """Wraps NeighbourhoodType switch method"""
        return self._neigh.switch(
            path=self._path, distances=distances, how=how, exclude=exclude
        )

    @property
    def _last_switch(self) -> tuple:
        return self._neigh.last_switch

    @property
    def _last_switch_comment(self) -> str:
        return self._neigh.last_switch_comment

    def __str__(self) -> str:
        mes = super().__str__()
        mes += f"""\nn_iter: {self._n_iter}\n"""
        return mes
