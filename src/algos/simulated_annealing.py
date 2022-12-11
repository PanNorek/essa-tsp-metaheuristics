from .swapping_algorithm import SwappingAlgorithm
import pandas as pd
from typing import Union, Callable, List
import random
import math

ALPHA = 0.9


def reduce(temp: float, alpha: float = ALPHA):
    return alpha * temp


def slowly_descend(temp: float, alpha: float = ALPHA) -> float:
    return temp/(1 + alpha * temp)


class SimulatedAnnealing(SwappingAlgorithm):
    """
    Simulated Annealing Algorithm
    
    Parameters
    ----------
    temp: int
        initial temperature
    alpha: float
        cooling factor
    reduce_func: Callable
        function to reduce temperature, default is reduce
    neigh_type: str
        type of neighbourhood, default is None
    n_iter: int
        number of iterations, default is 30
    verbose: bool
        print progress, default is False
    """

    NAME = "SIMULATED ANNEALING"

    def __init__(
        self,
        temp: int,
        alpha: float = ALPHA,
        reduce_func: Callable = reduce,
        neigh_type: str = None,
        n_iter: int = 30,
        verbose: bool = False,
    ) -> None:

        super().__init__(neigh_type=neigh_type, n_iter=n_iter, verbose=verbose)
        self._reduce_func = reduce_func
        self._alpha = alpha
        self._temp = temp

    def _iterate_steps(
        self, distances: pd.DataFrame, swaps: List[tuple]
    ) -> Union[int, None]:
        # start new iteration
        self._i += 1
        # find radom neighbouring solution and its distance
        swap, distance = self._find_random_swap(swaps=swaps, distances=distances)
        # distance gain
        gain = distance - self.history[-1]
        if gain < 0:
            # if distance is shorter, swap elements
            self._path = self._swap_elements(swap=swap)
            self.history.append(distance)
        else:
            rand = random.random()
            exp = math.exp(-gain / self._temp)
            if rand < exp:
                # if distance is longer but random [0,1] < exp(-DE/temp) swap elements
                self._path = self._swap_elements(swap=swap)
                self.history.append(distance)
            else:
                if self._verbose:
                    print(f"step {self._i}: path rejected")

        # reduce temperature
        self._temp = self._reduce_func(temp=self._temp, alpha=self._alpha)
        if self._verbose:
            print(f"swap: {swap} - gain: {gain}")

        return self.history[-1], swap

    def _find_random_swap(
        self, swaps: List[tuple], distances: pd.DataFrame
    ) -> pd.Series:
        # purely random in this algorithm
        swap = random.choice(swaps)
        new_path = self._swap_elements(swap=swap)
        path_distance = self._get_path_distance(distances=distances, path=new_path)
        return swap, path_distance
