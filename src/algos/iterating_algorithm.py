from abc import abstractmethod
from typing import Union
import pandas as pd
from .algorithm import TSPAlgorithm
from ..utils import (
    StopAlgorithm,
    Result,
    solve_it
)


class IteratingAlgorithm(TSPAlgorithm):
    """Swapping Algorithm"""

    DEFAULT_ITERS = 30
    DISTANCE = "distance"
    SWITCH = "switch"

    def __init__(self,
                 neigh_type: str = "swap",
                 n_iter: int = DEFAULT_ITERS,
                 verbose: bool = False,
                 ) -> None:
        super().__init__(
            neigh_type=neigh_type,
            verbose=verbose
        )
        self._n_iter = n_iter
        # current iteration
        self._i = 0
        self.history = []

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
        distance = self._get_path_distance(path=self._path, distances=distances)
        # list of distances at i iteration
        self.history = [distance]

        if self._verbose:
            print(f"--- {self.NAME} ---")
            print(f"step {self._i}: distance: {self.history[-1]}")

        self._iterate(distances=distances)

        # return result object
        return Result(
            algorithm=self,
            path=self._path,
            distance=self.history[-1],
            distance_history=self.history,
        )

    def _iterate(self, distances: pd.DataFrame,) -> None:
        for _ in range(self._n_iter):
            try:
                self._run_iteration(distances=distances)
            except StopAlgorithm as exc:
                if self._verbose:
                    print(exc.message)
                break

    @abstractmethod
    def _run_iteration(self, distances: pd.DataFrame) -> None:
        """Runs operations in a single interation

        Params:
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns

        It saves all information in attributes
        """
        pass

    def _next_iter(self) -> None:
        # start new iteration
        self._i += 1

    def _reset_iter(self) -> None:
        self._i = 0
        self.history = []

    def _setup_start(self,
                     distances: pd.DataFrame,
                     random_seed: Union[int, None] = None,
                     start_order: Union[list, None] = None
                     ) -> list:
        self._reset_iter()
        return super()._setup_start(
            distances=distances,
            random_seed=random_seed,
            start_order=start_order
        )

    def __str__(self) -> str:
        mes = super().__str__()
        mes += f"""\nn_iter: {self._n_iter}\n"""
        return mes
