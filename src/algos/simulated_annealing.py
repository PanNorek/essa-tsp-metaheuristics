from .swapping_algorithm import SwappingAlgorithm
from ..utils import time_it
import pandas as pd
from typing import Union, Callable, List
import random
import math

ALPHA = 0.1


def reduce(temp: float, alpha: float = ALPHA):
    return alpha * temp


@time_it
class SimulatedAnnealing(SwappingAlgorithm):
    """ Simulated Annealing Algorithm """
    NAME = 'SIMULATED ANNEALING'

    def __init__(self,
                 temp: int,
                 alpha: float = ALPHA,
                 reduce_func: Callable = reduce,
                 neigh_type: str = None,
                 n_iter: int = 30,
                 verbose: bool = False
                 ) -> None:

        super().__init__(neigh_type=neigh_type,
                         n_iter=n_iter,
                         verbose=verbose)
        self._reduce_func = reduce_func
        self._aplha = alpha
        self._temp = temp

    def _iterate_steps(self,
                       distances: pd.DataFrame,
                       swaps: List[tuple]
                       ) -> Union[int, None]:

        distance, swap = super()._iterate_steps(distances=distances,
                                                swaps=swaps)
        # distance gain
        gain = self.history[-1] - distance
        # break condition
        if gain > 0:
            # adding new best distance to distances history
            self.history.append(distance)
        else:
            rand = random.random()
            exp = math.exp(gain/self._temp)
            if rand < exp:
                # adding new best distance to distances history
                self.history.append(distance)
            else:
                if self._verbose:
                    print(f'swap: {swap} - gain: {gain}')
                    print(f'step {self._i}: path rejected')
                # return the same optimal distance
                return self.history[-1], None

        # reduce temperature
        self._temp = self._reduce_func(self._temp)

        if self._verbose:
            print(f'swap: {swap} - gain: {gain}')
            print(f'step {self._i}: distance: {distance}')

        return distance, swap

    def _find_best_swap(self,
                        swaps: List[tuple],
                        distances: pd.DataFrame
                        ) -> pd.Series:
        # purely random in this algorithm
        swap = random.choice(swaps)
        new_path = self._swap_elements(swap=swap)
        path_distance = self._get_path_distance(distances=distances,
                                                path=new_path)
        return swap, path_distance
