import math
import random
from typing import Union, Callable
import pandas as pd
from .switching_algorithm import SwitchingAlgorithm
from .iterating_algorithm import IteratingAlgorithm


ALPHA = 0.9


def reduce(temp: float, alpha: float = ALPHA):
    """Reduces temperature by multiplying it by alpha"""
    return alpha * temp


def slowly_descend(temp: float, alpha: float = ALPHA) -> float:
    """Reduces temperature by dividing it by 1 + alpha * temp"""
    return temp / (1 + alpha * temp)


class SimulatedAnnealing(SwitchingAlgorithm, IteratingAlgorithm):
    """
    Simulated Annealing Algorithm

    Parameters
    ----------
    temp: int
        initial temperature
        The temperature progressively decreases from an initial positive value to zero.
    alpha: float
        cooling factor
        The cooling factor is a number between 0 and 1 that controls
        the rate at which the temperature decreases.
    reduce_func: Callable
        function to reduce temperature, default is reduce
        Available options: reduce, slowly_descend
    neigh_type: str
        type of neighbourhood, default is None
        Available options: 'swap', 'insert', 'inversion'
    n_iter: int
        number of iterations, default is 100
    verbose: bool
        print progress, default is False
    """
    DEFAULT_ITERS = 100
    NAME = "SIMULATED ANNEALING"
    _REDUCE_FUNC = {
        "reduce": reduce,
        "descend": slowly_descend
    }

    def __init__(self,
                 # TODO: check empirically xd
                 temp: int = 100,
                 alpha: float = ALPHA,
                 reduce_func: Union[Callable, str] = "reduce",
                 neigh_type: str = "swap",
                 n_iter: int = DEFAULT_ITERS,
                 verbose: bool = False,
                 ) -> None:
        super().__init__(
            neigh_type=neigh_type,
            n_iter=n_iter,
            verbose=verbose
        )
        if not callable(reduce_func):
            assert (
                reduce_func in self._REDUCE_FUNC
            ), f"reduce_func must be one of {list(self._REDUCE_FUNC.keys())} or a function"
            reduce_func = self._REDUCE_FUNC[reduce_func]
        self._reduce_func: Callable = reduce_func
        self._alpha = alpha
        self._temp = self._start_temp = temp

    def _run_iteration(self, distances: pd.DataFrame) -> None:
        self._next_iter()
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
            print(f"temperature reduced to {self._temp}")
            print(f"switch: {self._last_switch_comment} - gain: {diff}")

    def __str__(self) -> str:
        mes = super().__str__()
        mes += f"""reduction function: {self._reduce_func.__name__}\n\
            alpha: {self._alpha}\n\
            start temperatute: {self._start_temp}"""
        return mes
