from .algorithm import Algorithm
from ..utils import time_it
from typing import Tuple
import pandas as pd

@time_it
class NearestNeighbour(Algorithm):
    " Nearest Neighbour Algorithm"
    def __init__(self, neigh_type: str = None):
        super().__init__(neigh_type)

    def solve(self, distance_matrix: pd.DataFrame, start: int) -> Tuple[int, str]:
        """Solve TSP problem with Nearest Neighbour Algorithm

        Args:
            distance_matrix (pd.DataFrame): Distance matrix
            start (int): Start node
        
        Returns:
            distance (int): Total distance
            path (str): Path 
        """
        end = start
        unvisited = list(range(1, len(distance_matrix) + 1))
        unvisited.remove(start)
        distance = 0
        path = str(start)

        while unvisited:

            current = end

            nearest_distance = distance_matrix.iloc[current-1, unvisited[0]-1]

            for neighbour in unvisited:

                try:
                    if distance_matrix.iloc[current-1, unvisited[neighbour]-1] < nearest_distance:
                        current = neighbour
                except IndexError:
                    pass

            try:
                unvisited.remove(current)
            except ValueError:
                return (distance, path)
            path += '-' + str(current)
            distance += nearest_distance
            end = current
