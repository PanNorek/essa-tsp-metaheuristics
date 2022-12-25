import random
from abc import ABC, abstractmethod
from typing import Union, Iterable
import pandas as pd
import numpy as np
from ..utils import (
    Inversion,
    Swap,
    Insertion,
    NeighbourhoodType,
    Result,
    get_path_distance
)


class Algorithm(ABC):
    """Traveling Salesman Problem (TSP) solver"""

    NAME = ""
    _NEIGHBOURHOOD_TYPES = {
        "swap": Swap,
        "inversion": Inversion,
        "insertion": Insertion,
    }

    def __init__(self,
                 neigh_type: str = "swap",
                 verbose: bool = False
                 ) -> None:
        """
        Params:
            neigh_type: str
                Type of neighbourhood used in algorithm
            verbose: bool
                If True, prints information about algorithm progress
            inversion_window: Union[int, None]
                If neighbourhood type is inversion, this parameter
                determines how many cities are swapped in one step
        """
        self._verbose = verbose
        self._path = []
        self.history = []
        neigh = self._NEIGHBOURHOOD_TYPES.get(neigh_type)
        assert (
            neigh
        ), f"neigh_type must be one of {list(self._NEIGHBOURHOOD_TYPES.keys())}"
        self._neigh_type = neigh
        self._neigh = None

    def solve(self,
              distances: pd.DataFrame,
              random_seed: Union[int, None] = None,
              start_order: Union[list, None] = None
              ) -> Result:
        """
        Uses specific algorithm to solve Traveling Salesman Problem

        Params:
            distances: pd.DataFrame
                Matrix od distances between cities,
                cities numbers or id names as indices and columns
        """
        return self._solve(
            distances=distances,
            random_seed=random_seed,
            start_order=start_order
        )

    @abstractmethod
    def _solve(self,
               distances: pd.DataFrame,
               random_seed: Union[int, None] = None,
               start_order: Union[list, None] = None
               ) -> Result:
        pass

    def _setup_start(self,
                     distances: pd.DataFrame,
                     random_seed: Union[int, None] = None,
                     start_order: Union[list, None] = None
                     ) -> list:
        """
        Sets up a starting order for an algorithm
        """
        self._setup(distances=distances, random_seed=random_seed)
        path = self._set_start_order(
            distances=distances, start_order=start_order)
        return path

    def _setup(self,
               distances: pd.DataFrame,
               random_seed: Union[int, None] = None
               ) -> None:
        self._distance_matrix_check(distances=distances)
        self._set_random_seed(random_seed=random_seed)
        self._set_neighbourhood(distances=distances)

    def _set_neighbourhood(self, distances: pd.DataFrame) -> None:
        self._neigh: NeighbourhoodType = self._neigh_type(
            path_length=len(distances)
        )

    def _set_start_order(self,
                         distances: pd.DataFrame,
                         start_order: Union[list, None] = None
                         ) -> list:
        if start_order is not None:
            self._start_order_check(start_order=start_order, distances=distances)
            return start_order
        return self._get_random_start(indices=distances.index)

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

    def _get_path_distance(self, path: list, distances: pd.DataFrame) -> int:
        """Wraps get_path_distance function"""
        return get_path_distance(path=path, distances=distances)

    def _get_random_start(self, indices: Union[list, pd.Index]):
        # indices of the cities
        if isinstance(indices, pd.Index):
            indices = list(indices)
        # retrun random path
        return random.sample(indices, len(indices))

    def _set_random_seed(self, random_seed: int = None) -> None:
        # sets random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def _distance_matrix_check(self, distances: pd.DataFrame) -> None:
        mes = "indices and columns of distances matrix should be equal"
        assert distances.index.to_list() == distances.columns.to_list(), mes

    @property
    def best_path(self) -> list:
        """Returns the most optimal graph's path that was found"""
        return self._path

    def __str__(self) -> str:
        return f"{self.NAME}\nNeighbourhood type: {str(self._neigh)}"

    def __repr__(self) -> str:
        return str(self)
