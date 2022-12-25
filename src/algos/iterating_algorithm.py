from abc import abstractmethod
from typing import Union
import pandas as pd
from .algorithm import Algorithm


class IteratingAlgorithm(Algorithm):
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

    @abstractmethod
    def _iterate_steps(self, distances: pd.DataFrame) -> None:
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
