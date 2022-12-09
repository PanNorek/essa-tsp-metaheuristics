from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union


class Algorithm(ABC):
    """ Traveling Salesman Problem (TSP) solver """

    def __init__(self, neigh_type: str = None) -> None:
        self.neigh_type = neigh_type
        self._path = []

    @abstractmethod
    def solve(self, distances: Union[pd.DataFrame, np.ndarray]) -> int:
        """
        Uses specific algorithm to solve Traveling Salesman Problem

        Params:
            distances: pd.DataFrame | np.ndarray
                Matrix od distances between cities,
                cities numbers or id names as indices and columns
        """
        pass

    def _distance_matrix_check(self, distances: Union[pd.DataFrame, np.ndarray]) -> None:
        if not isinstance(distances, pd.DataFrame):
            return
        mes = "indices and columns of distances matrix should be equal"
        assert distances.index.to_list() == distances.columns.to_list(), mes

    @property
    def path(self) -> list:
        """ Returns the most optimal graph's path that was found """
        return self._path
