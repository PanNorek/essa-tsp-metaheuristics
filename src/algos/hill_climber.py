from .algorithm import Algorithm
from ..utils import time_it
from typing import Tuple, Union
import pandas as pd
import numpy as np
from multiprocessing import Pool


def unwrap_self_f(arg, **kwarg):
    return HillClimber.solve_multistart_parallel(*arg, **kwarg)


class HillClimber(Algorithm):
    " Hill Climber Algorithm"
    def __init__(self, neigh_type: str = None):
        super().__init__(neigh_type)

    @time_it
    def solve(self,
              distance_matrix: pd.DataFrame,
              starting_solution: Union[list, np.ndarray] = None,
              num_iter: int = 50,
              return_tuple: bool = False
              ) -> Union[Tuple[int, str], int]:
        """Solve TSP problem with Hill Climber Algorithm

            Args:
                distance_matrix (pd.DataFrame): Distance matrix
                starting_solution (int): Starting salesman path
                num_iter (int): Number of iterations

            Returns:
                distance (int): Total distance
                path (str): Salesman path
        """

        # checks if starting point was properly defined
        if starting_solution is None:
            starting_solution = distance_matrix.index.to_numpy()
            np.random.shuffle(starting_solution)

        # print(starting_solution)

        # TODO: add more sanity checks
        assert isinstance(starting_solution, (list, np.ndarray)), 'bad starting solution'

        # if you're checked first then you're always the best for a while
        best_solution = starting_solution
        for i in range(1, num_iter+1):

            best_solution = self._hill_climber_step(best_solution, distance_matrix)
            if i % 50 == 0:
                print(f"Iteration {i}")
        if return_tuple:
            return (self._compute_distance(best_solution, distance_matrix), best_solution)

        return self._compute_distance(best_solution, distance_matrix)

    def solve_multistart(self,
                         distance_matrix: pd.DataFrame,
                         num_iter: int = 20,
                         num_starts: int = 10
                         ) -> Tuple[int, str]:
        """Solve TSP problem with Hill Climber Algorithm with multiple starts

        Args:
            distance_matrix (pd.DataFrame): Distance matrix
            num_iter (int): Number of iterations
            num_starts (int): Number of starts

        Returns:
            distance (int): Total distance
            path (str): Salesman path
        """

        # TODO: add multiprocessing
        results = []
        for i in range(num_starts):
            results.append(self.solve(distance_matrix,
                                      num_iter=num_iter,
                                      return_tuple=True))
            print(f"Start {i}")

        return min(results, key=lambda x: x[0])
        # return results

    def solve_multistart_parallel(self,
                                  distance_matrix: pd.DataFrame,
                                  num_iter: int = 20,
                                  num_starts: int = 0
                                  ) -> Tuple[int, str]:
        """Solve TSP problem with Hill Climber Algorithm with multiple starts

        Args:
            distance_matrix (pd.DataFrame): Distance matrix
            num_iter (int): Number of iterations
            num_starts (int): Number of starts

        Returns:
            distance (int): Total distance
            path (str): Salesman path
        """
        pool = Pool(processes=2)
        pool.map(unwrap_self_f, [(self, distance_matrix, num_iter, num_starts)])

    def _compute_distance(self, path: np.array, distance_matrix: pd.DataFrame) -> int:
        """Compute total distance of given path

        Args:
            distance_matrix (pd.DataFrame): Distance matrix
            path (list): Salesman path

        Returns:
            distance (int): Total distance
        """

        path = np.append(path, path[0])
        return np.sum((distance_matrix.loc[x, y] for x, y in zip(path, path[1:])))

    def _find_neighbours(self, path: np.array) -> np.ndarray:
        """Find neighbours of given path

        Args:
            path (np.array): Salesman path

        Returns:
            neighbours (np.ndarray): List of neighbours
        """
        neighbours = [path]
        for i in range(len(path) - 1):
            for j in range(i + 1, len(path)):
                neighbours.append(self._swap(path, i, j))

        return np.array(neighbours)

    def _swap(self, path: np.array, i: int, j: int) -> np.array:
        """Swap two elements in given path

        Args:
            path (np.array): Salesman path
            i (int): Index of first element
            j (int): Index of second element

        Returns:
            path (np.array): Path with swapped elements
        """
        path = path.copy()
        path[i], path[j] = path[j], path[i]

        return path

    def _find_best_neighbour(self, neighbours: np.ndarray, **kwargs) -> np.array:
        """Find best neighbour of given path

        Args:
            neighbours (np.array): List of neighbours

        Returns:
            best_neighbour (np.array): Best neighbour
        """
        return neighbours[np.argmin([self._compute_distance(neighbour, **kwargs)
                                     for neighbour in neighbours])]

    def get_distances(self, neighbours: np.array, **kwargs) -> pd.Series:
        """Get distances of given neighbours

        Args:
            neighbours (np.array): List of neighbours

        Returns:
            distances (np.array): List of distances
        """
        return pd.Series({self._compute_distance(neighbour, **kwargs): neighbour
                          for neighbour in neighbours})

    def _hill_climber_step(self, path: np.array, distance_matrix: pd.DataFrame) -> np.array:
        """Perform one step of hill climber algorithm

        Args:
            path (np.array): Salesman path
            distance_matrix (pd.DataFrame): Distance matrix

        Returns:
            path (np.array): Salesman path
        """
        # Find solutions neighbours
        neighbours = self._find_neighbours(path)
        # Find best neighbour
        best_neighbour = self._find_best_neighbour(neighbours, distance_matrix=distance_matrix)
        # Check if best neighbour is better than current solution
        if (self._compute_distance(path, distance_matrix) > self._compute_distance(best_neighbour, distance_matrix)):
            return best_neighbour
        else:
            return path
