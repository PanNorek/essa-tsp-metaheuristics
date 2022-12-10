from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import random
from typing import Union, List


class Algorithm(ABC):
    """ Traveling Salesman Problem (TSP) solver """
    NAME = None

    def __init__(self,
                 neigh_type: str = None,
                 verbose: bool = False
                 ) -> None:
        self.neigh_type = neigh_type
        self._verbose = verbose
        self._path = []

    @abstractmethod
    def solve(self, distances: pd.DataFrame) -> int:
        """
        Uses specific algorithm to solve Traveling Salesman Problem

        Params:
            distances: pd.DataFrame | np.ndarray
                Matrix od distances between cities,
                cities numbers or id names as indices and columns
        """
        pass

    def _get_random_path(self, indices: Union[list, pd.Index]):
        # indices of the cities
        if isinstance(indices, pd.Index):
            indices = list(indices)
        # retrun random path
        return random.sample(indices, len(indices))

    def _set_random_seed(self, random_seed: int = None) -> None:
        # sets random seed
        random.seed(random_seed)

    def _distance_matrix_check(self, distances: pd.DataFrame) -> None:
        if not isinstance(distances, pd.DataFrame):
            return
        mes = "indices and columns of distances matrix should be equal"
        assert distances.index.to_list() == distances.columns.to_list(), mes

    def _get_path_distance(self, distances: pd.DataFrame, path: list) -> int:
        """ Calculate distance of the path based on distances matrix """
        path_length = sum([distances.loc[x, y] for x, y in zip(path, path[1:])])
        # add distance back to the starting point
        path_length += distances.loc[path[0], path[-1]]
        return path_length

    @property
    def path(self) -> list:
        """ Returns the most optimal graph's path that was found """
        return self._path
