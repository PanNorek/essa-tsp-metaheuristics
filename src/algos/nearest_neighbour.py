from .algorithm import Algorithm
from ..utils import time_it
import pandas as pd
import numpy as np
from typing import Union
import random


@time_it
class NearestNeighbour(Algorithm):
    """ Nearest Neighbour Algorithm """

    def solve(self,
              distances: Union[pd.DataFrame, np.ndarray],
              start: Union[int, None] = None
              ) -> int:
        # if starting point is None, choose it randomly from indices
        if start is None:
            start = random.choice(list(distances.index))
        # checks if starting point was properly defined
        assert start in list(distances.index), 'starting point not in distance matix'
        # checks if columns are the equal to indices
        self._distance_matrix_check(distances=distances)
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
        return distance
