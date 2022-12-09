from .algorithm import Algorithm
from ..utils import Queue
import pandas as pd
from typing import Union, List
import random


class TabuSearch(Algorithm):
    """ Tabu Search Algorithm """
    DISTANCE = 'distance'
    SWAP = 'swap'

    def __init__(self,
                 tabu_length: int = 3,
                 neigh_type: str = None,
                 n_iter: int = 100,
                 verbose: bool = False
                 ) -> None:
        super().__init__(neigh_type=neigh_type)
        self._tabu_length = tabu_length
        self._verbose = verbose
        self._n_iter = n_iter
        # current iteration
        self.__i = 0
        # initializing tabu list queue
        self._tabu_list = Queue(length=self._tabu_length)

    def solve(self,
              distances: pd.DataFrame,
              random_seed: int = None
              ) -> int:
        # sets random seed
        random.seed(random_seed)
        # indices of the cities
        indices = list(distances.index)
        # random path at the beginning
        self._path = random.sample(indices, len(indices))
        # getting all posible swaps
        swaps = self._get_all_swaps(indices=indices)
        # distance of the path at the beginning
        distance = self._get_path_distance(distances=distances,
                                           path=self._path)
        if self._verbose:
            print('--- TABU SEARCH ---')
            print(f'step {self.__i}: distance: {distance}')

        # list of distances at i iteration
        self.history = [distance]

        for _ in range(self._n_iter):
            # best neighbouring solutions - minimizes distance at this moment
            best_swap, best_distance = self._get_best_swap(swaps=swaps,
                                                           distances=distances)
            # new path that minimizes distance
            self._path = self._swap_elements(swap=best_swap)
            # removing first swap from tabu list only if list is full
            self._tabu_list.dequeue()
            # adding new swap to tabu list
            self._tabu_list.enqueue(best_swap)
            # adding new best distance to distances history
            self.history.append(best_distance)
            # start new iteration
            self.__i += 1

            if self._verbose:
                print(f'best swap: {best_swap} - gain: {best_distance - distance}')
                print(f'step {self.__i}: distance: {distance}')
                print(f'tabu list: {self._tabu_list}')

        return distance

    def _get_all_swaps(self, indices: list) -> List[tuple]:
        """ Returns all possible swaps of indices """
        index = len(indices) + 1
        # unique combination
        swaps = [(x, y) for x in range(1, index)
                 for y in range(1, index)
                 if (x != y) and (y > x)]
        return swaps

    def _get_best_swap(self, swaps: List[tuple], distances: pd.DataFrame) -> pd.DataFrame:
        # new paths with swapped order
        new_paths = list(map(lambda x: self._swap_elements(swap=x), swaps))
        # distances of all possible new paths
        paths_distances = list(map(lambda x: self._get_path_distance(
            distances=distances, path=x), new_paths))
        # DataFrame of all swaps and distances of formed paths
        distances_df = pd.DataFrame({self.SWAP: swaps, self.DISTANCE: paths_distances})
        # sorting distances in ascending order
        distances_df = distances_df.sort_values('distance')
        # dropping all swaps that are in tabu list
        distances_df = distances_df[~distances_df[self.SWAP].isin(self._tabu_list)]
        # taking row of the best swaps
        best_swap = distances_df.iloc[0]
        return best_swap

    def _swap_elements(self, swap: tuple) -> list:
        """ Returns copy of the current path with swapped indices """
        path = self._path.copy()
        pos1 = path.index(swap[0])
        pos2 = path.index(swap[1])
        path[pos1], path[pos2] = path[pos2], path[pos1]
        return path

    def _get_path_distance(self, distances: pd.DataFrame, path: list) -> int:
        """ Calculate distance of the path based on distances matrix """
        path_length = sum([distances.loc[x, y] for x, y in zip(path, path[1:])])
        # add distance back to the starting point
        path_length += distances.loc[path[0], path[-1]]
        return path_length
