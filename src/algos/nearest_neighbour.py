import random
from typing import Union, Iterable
import pandas as pd
from .algorithm import Algorithm
from ..utils import solve_it, Result


class NearestNeighbour(Algorithm):
    """Nearest Neighbour Algorithm"""

    NAME = "NEAREST NEIGHBOUR"

    @solve_it
    def _solve(self,
               distances: pd.DataFrame,
               start_order: Union[int, None] = None,
               random_seed: Union[int, None] = None,
               ) -> int:
        start_order = self._setup_start(
            distances=distances,
            random_seed=random_seed,
            start_order=start_order
        )
        # sets unvisited list to all cities
        unvisited = list(distances.index)
        # first visited city is starting point
        unvisited.remove(start_order)
        # distance at the beginning
        distance = 0
        # _path attribute is the travelled path between cities
        self._path.append(start_order)
        # while there are still cities to visit
        while unvisited:
            # city salesman is currently in
            current = self._path[-1]
            # all possible distances between current city and unvisited ones
            possibilities = distances.loc[current, unvisited]
            # index of the nearest city
            nearest_city = possibilities.idxmin()
            # distance between current and nearest cities
            distance += possibilities.min()
            # goes to the nearest city
            self._path.append(nearest_city)
            # removes newly visited city from unvisited list
            unvisited.remove(nearest_city)
        # return to the first city
        distance += distances.loc[self._path[-1], start_order]
        result = Result(algorithm=self, path=self._path, distance=distance)
        return result

    def _get_random_start(self, indices: Union[list, pd.Index]):
        # indices of the cities
        if isinstance(indices, pd.Index):
            indices = list(indices)
        # return random city
        return random.choice(indices)

    def _start_order_check(self,
                           start_order: list,
                           distances: pd.DataFrame
                           ) -> None:
        assert not isinstance(start_order, list), 'start_order must not be a list for NN'
        assert start_order in list(distances.index), 'starting point not in distance matix'

    def _set_neighbourhood(self, distances: pd.DataFrame) -> None:
        self._neigh = None

    def __str__(self) -> str:
        return f"{self.NAME}\n"
