from abc import abstractmethod
from typing import Union, Iterable
import pandas as pd
from .algorithm import Algorithm
from ..utils import (
    StopAlgorithm,
    Result,
    solve_it,
    get_path_distance,
    Queue
)


class SwappingAlgorithm(Algorithm):
    """Swapping Algorithm"""

    DISTANCE = "distance"
    SWITCH = "switch"

    def __init__(self,
                 neigh_type: str = "swap",
                 n_iter: int = 30,
                 verbose: bool = False,
                 ) -> None:
        super().__init__(
            neigh_type=neigh_type,
            verbose=verbose
        )
        self._n_iter = n_iter
        # current iteration
        self._i = 0

    @solve_it
    def _solve(self,
               distances: pd.DataFrame,
               random_seed: Union[int, None] = None,
               start_order: Union[list, None] = None
               ) -> Result:
        super()._solve(distances=distances, random_seed=random_seed)
        if start_order is not None:
            self._start_order_check(start_order=start_order, distances=distances)

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

    def _start_order_check(self,
                           start_order: list,
                           distances: pd.DataFrame
                           ) -> None:
        assert isinstance(start_order, Iterable), 'start_order must be iterable'
        assert (
            len(start_order) == len(distances)
        ), f'Expected {len(distances)} elements, got {len(start_order)}'
        assert (
            all(index in distances.index.to_list() for index in start_order)
        ), 'elements of start_order must allign with distance matrix indices'
        assert len(set(start_order)) == len(start_order), 'elements in start_order must be unique'

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
