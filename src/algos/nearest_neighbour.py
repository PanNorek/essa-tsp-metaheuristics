import random
from typing import Union, Any
import pandas as pd
from .algorithm import TSPAlgorithm
from ..utils import solve_it, Result


class NearestNeighbour(TSPAlgorithm):
    """
    Nearest Neighbour Algorithm

    One of the simplest one.
    The nearest neighbour algorithm was one of the first
    algorithms used to solve the travelling salesman problem approximately.
    In that problem, the salesman starts at a random city and repeatedly visits
    the nearest city until all have been visited.
    The algorithm quickly yields a short tour, but usually not the optimal one.

    For more information checks out wikipedia page
    https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm
    """

    NAME = "NEAREST NEIGHBOUR"

    @solve_it
    def _solve(self,
               distances: pd.DataFrame,
               random_seed: Union[int, None] = None,
               start_order: Union[int, None] = None
               ) -> int:
        # specific implementation for NN algorithm
        # gets the starting point city
        start_order = self._setup_start(
            distances=distances,
            random_seed=random_seed,
            start_order=start_order
        )
        # _path attribute is the travelled path between cities
        # the first element is city chosen as starting point
        # it is also the city traveling salesman has to end up in
        self._path = [start_order]
        # sets unvisited list to all cities
        unvisited = list(distances.index)
        # first visited city is starting point
        unvisited.remove(start_order)
        # while there are still cities to visit
        while unvisited:
            # all possible distances between current city and unvisited ones
            possibilities = distances.loc[self._path[-1], unvisited]
            # index of the nearest city
            nearest_city = possibilities.idxmin()
            # salesman goes to the nearest city
            self._path.append(nearest_city)
            # removes newly visited city from unvisited list
            unvisited.remove(nearest_city)

        # calculates the final distance TS has to traverse
        distance = self._get_path_distance(path=self._path, distances=distances)
        # returns Result object
        return Result(
            algorithm=self,
            path=self._path,
            distance=distance
        )

    def _get_random_start(self, indices: Union[list, pd.Index]) -> Any:
        """
        Chooses random starting point

        Params:
            indices: list | pd.Index
                Cities to be visited by the salesman
        Returns:
            One randomly chosen element from indices representing
            starting city

        Overrides method of TSPAlgorithm superclass giving
        implementation specific to NN
        """
        # indices of the cities
        if isinstance(indices, pd.Index):
            indices = list(indices)
        # return random city
        return random.choice(indices)

    def _start_order_check(self,
                           start_order: Any,
                           distances: pd.DataFrame
                           ) -> None:
        """
        Runs the series of checks to assert that starting path provided by
        user is correct and can be used in solve method

        Params:
            start_order: list | None
                Order from which algorithm starts solving problem,
                if None, order will be chosen randomly
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns

        Basic checks include:
            checking that start_path is not a list
            checking whether start_path is in indices

        Overrides method of TSPAlgorithm superclass giving
        implementation specific to NN. This algorithm needs
        an element representing a city as a starting point
        """
        assert not isinstance(start_order, list), 'start_order must not be a list for NN'
        assert start_order in list(distances.index), 'starting point not in distance matix'

    def _set_neighbourhood(self, distances: pd.DataFrame) -> None:
        """
        Returns None as NN doesn't need to search neighbouring solutions space

        Params:
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns
        """
        self._neigh = None

    def __str__(self) -> str:
        return f"{self.NAME}\n"
