import random
from typing import Union
import pandas as pd
from .algorithm import Algorithm
from ..utils import solve_it, Result


class NearestNeighbour(Algorithm):
    """Nearest Neighbour Algorithm"""

    NAME = "NEAREST NEIGHBOUR"

    @solve_it
    def solve(self,
              distances: pd.DataFrame,
              start: Union[int, None] = None,
              random_seed: Union[int, None] = None,
              ) -> int:
        super().solve(distances=distances, random_seed=random_seed)
        # if starting point is None, choose it randomly from indices
        if start is None:
            start = random.choice(list(distances.index))
        # checks if starting point was properly defined
        assert start in list(distances.index), "starting point not in distance matix"
        # sets unvisited list to all cities
        unvisited = list(distances.index)
        # first visited city is starting point
        unvisited.remove(start)
        # distance at the beginning
        distance = 0
        # _path attribute is the travelled path between cities
        self._path.append(start)
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
        distance += distances.loc[self._path[-1], start]
        result = Result(algorithm=self, path=self._path, best_distance=distance)
        return result

    def __str__(self) -> str:
        return f"{self.NAME}\n"