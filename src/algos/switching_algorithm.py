from abc import abstractmethod
from typing import Union
import pandas as pd
from .algorithm import TSPHeuristicAlgorithm
from ..utils import Queue


class SwitchingAlgorithm(TSPHeuristicAlgorithm):
    """Swapping Algorithm"""

    def _switch(
        self,
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
