from .algorithm import Algorithm
from typing import Union, List
import pandas as pd
from ..utils import StopAlgorithm


class SwappingAlgorithm(Algorithm):
    DISTANCE = 'distance'
    SWAP = 'swap'

    def __init__(self,
                 neigh_type: str = None,
                 n_iter: int = 30,
                 verbose: bool = False
                 ) -> None:
        super().__init__(neigh_type=neigh_type, verbose=verbose)
        self._n_iter = n_iter
        # current iteration
        self._i = 0

    def solve(self,
              distances: pd.DataFrame,
              random_seed: int = None,
              ) -> int:
        # checks if columns are the equal to indices
        self._distance_matrix_check(distances=distances)
        self._set_random_seed(random_seed=random_seed)
        # random path at the beginning
        self._path = self._get_random_path(indices=distances.index)
        # getting all posible swaps
        swaps = self._get_all_swaps(indices=distances.index)
        # distance of the path at the beginning
        distance = self._get_path_distance(distances=distances,
                                           path=self._path)
        # list of distances at i iteration
        self.history = [distance]

        if self._verbose:
            print(f'--- {self.NAME} ---')
            print(f'step {self._i}: distance: {distance}')

        for _ in range(self._n_iter):
            try:
                best_distance, _ = self._iterate_steps(distances=distances,
                                                       swaps=swaps)
            except StopAlgorithm as exc:
                if self._verbose:
                    print(exc.message)
                break
        return best_distance

    def _iterate_steps(self,
                       distances: pd.DataFrame,
                       swaps: List[tuple]
                       ) -> Union[int, None]:
        best_swap, best_distance = self._find_best_swap(swaps=swaps,
                                                        distances=distances)
        # new path that minimizes distance
        self._path = self._swap_elements(swap=best_swap)
        # adding new best distance to distances history
        self.history.append(best_distance)
        # start new iteration
        self._i += 1
        return best_distance, best_swap

    def _swap_elements(self, swap: tuple) -> list:
        """ Returns copy of the current path with swapped indices """
        path = self._path.copy()
        pos1 = path.index(swap[0])
        pos2 = path.index(swap[1])
        path[pos1], path[pos2] = path[pos2], path[pos1]
        return path

    def _find_best_swap(self,
                        swaps: List[tuple],
                        distances: pd.DataFrame
                        ) -> pd.Series:
        distances_df = self._get_swaps_df(swaps=swaps, distances=distances)
        # taking row of the best swaps
        best_swap = distances_df.iloc[0]
        return best_swap

    def _get_swaps_df(self,
                      swaps: List[tuple],
                      distances: pd.DataFrame
                      ) -> pd.DataFrame:
        # new paths with swapped order
        new_paths = list(map(lambda x: self._swap_elements(swap=x), swaps))
        # distances of all possible new paths
        paths_distances = list(map(lambda x: self._get_path_distance(
            distances=distances, path=x), new_paths))
        # DataFrame of all swaps and distances of formed paths
        distances_df = pd.DataFrame({self.SWAP: swaps, self.DISTANCE: paths_distances})
        # sorting distances in ascending order
        distances_df = distances_df.sort_values(self.DISTANCE)
        return distances_df

    def _get_all_swaps(self, indices: list) -> List[tuple]:
        """ Returns all possible swaps of indices """
        index = len(indices) + 1
        # unique combination
        swaps = [(x, y) for x in range(1, index)
                 for y in range(1, index)
                 if (x != y) and (y > x)]
        return swaps
