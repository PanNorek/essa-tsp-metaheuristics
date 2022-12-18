import math
import random
from typing import Union, Callable, List
import pandas as pd
from .swapping_algorithm import SwappingAlgorithm


ALPHA = 0.9


def reduce(temp: float, alpha: float = ALPHA):
    """Reduces temperature by multiplying it by alpha"""
    return alpha * temp


def slowly_descend(temp: float, alpha: float = ALPHA) -> float:
    """Reduces temperature by dividing it by 1 + alpha * temp"""
    return temp / (1 + alpha * temp)


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
    _REDUCE_FUNC = {
        "reduce": reduce,
        "descend": slowly_descend
    }

    def __init__(self,
                 temp: int,
                 alpha: float = ALPHA,
                 reduce_func: Callable | str = "reduce",
                 neigh_type: str = "swap",
                 n_iter: int = 30,
                 verbose: bool = False,
                 inversion_window: int | None = None
                 ) -> None:
        super().__init__(neigh_type=neigh_type,
                         n_iter=n_iter,
                         verbose=verbose,
                         inversion_window=inversion_window)
        if not callable(reduce_func):
            assert (
                reduce_func in self._REDUCE_FUNC
            ), f"reduce_func must be one of {list(self._REDUCE_FUNC.keys())} or a function"
            reduce_func = self._REDUCE_FUNC[reduce_func]
        self._reduce_func: Callable = reduce_func
        self._alpha = alpha
        self._temp = temp

    def _iterate_steps(self, distances: pd.DataFrame) -> None:
        # start new iteration
        self._i += 1
        # find radom neighbouring solution and its distance
        new_path = self._switch(distances=distances, how='random')
        distance = self._get_path_distance(path=new_path, distances=distances)
        # distance gain
        diff = distance - self.history[-1]
        if diff < 0:
            # if distance is shorter, swap elements
            self._path = new_path
            self.history.append(distance)
        else:
            rand = random.random()
            exp = math.exp(-diff / self._temp)
            if rand < exp:
                # if distance is longer but random [0,1] < exp(-DE/temp) swap elements
                self._path = new_path
                self.history.append(distance)
            elif self._verbose:
                print(f"step {self._i}: path rejected")

        # reduce temperature
        self._temp = self._reduce_func(temp=self._temp, alpha=self._alpha)
        if self._verbose:
            print(f"switch: {self._last_switch_comment} - gain: {diff}")
