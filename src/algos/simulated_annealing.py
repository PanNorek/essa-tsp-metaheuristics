from math import exp, isclose
import random
from typing import Union, Callable
import pandas as pd
from .iterating_algorithm import IteratingAlgorithm
from ..utils import StopAlgorithm


ALPHA = 0.9


def reduce(temp: float, alpha: float = ALPHA):
    """Reduces temperature by multiplying it by alpha"""
    return alpha * temp


def slowly_descend(temp: float, alpha: float = ALPHA) -> float:
    """Reduces temperature by dividing it by 1 + alpha * temp"""
    return temp / (1 + alpha * temp)


class SimulatedAnnealing(IteratingAlgorithm):
    """
    Simulated Annealing Algorithm

    Methods:
        solve - used for solving PFSP problem

    Attributes:
        path_ - best order found by algorithm
        history - list of best orders from each iteration

    Implements:
        IteratingAlgorithm - provides method to facilitates and order
            iterative approach to solving PFSP with use of object attributes

    Simulated annealing (SA) is a probabilistic technique for approximating
    the global optimum of a given function. Specifically, it is a metaheuristic
    to approximate global optimization in a large search space for an optimization problem.
    This notion of slow cooling implemented in the simulated annealing algorithm is
    interpreted as a slow decrease in the probability of accepting worse solutions
    as the solution space is explored.
    Accepting worse solutions allows for a more extensive search for the global optimal solution
    The temperature progressively decreases from an initial positive value to zero.
    At each time step, the algorithm randomly selects a solution close to the current one,
    measures its quality, and moves to it according to the temperature-dependent
    probabilities of selecting better or worse solutions, which during the
    search respectively remain at 1 (or positive) and decrease toward zero.

    For more information check out:
    https://mathworld.wolfram.com/SimulatedAnnealing.html
    https://en.wikipedia.org/wiki/Simulated_annealing
    """

    DEFAULT_ITERS = 1000
    NAME = "SIMULATED ANNEALING"
    _REDUCE_FUNC = {"reduce": reduce, "descend": slowly_descend}

    def __init__(
        self,
        temp: int = 100,
        alpha: float = ALPHA,
        reduce_func: Union[Callable, str] = "reduce",
        stopping_tol: Union[float, None] = None,
        neigh_type: str = "swap",
        n_iter: int = DEFAULT_ITERS,
        verbose: bool = False,
    ) -> None:
        """
        Params:
            temp: int
                Initial temperature that
                progressively decreases from an initial positive value to zero.
            alpha: float
                The cooling factor is a number between 0 and 1 that controls
                the rate at which the temperature decreases.
            reduce_func: str | Callable
                function used to reduce temperature each iteration
            stopping_tol: float | None
                maximum absolute difference for considering temperature "close" to 0
                and stopping iteration, None by default
            neigh_type: str
                Type of neighbourhood used in algorithm
            n_iter: int
                Max number of iterations to run before finishing the algorithm
            verbose: bool
                If True, prints out information about algorithm progress

        temp:
            the greater the temperature the greater chance algorithm has
            to accept less optimal adjacent solution.
            Temperature decreases with each iteration in line with reduce_func

        alpha:
            depends on the reduction function

            In "reduce" value close to 1 means slowly decreasing temperature
            value close to 0 means sheer decrease in temperature

            In "slowly_descend" value close to 1 means rapidly decreasing temperature
            value close to 0 means slow decrease in temperature

            If reduce function parameter is a function, alpha is not applicable

        reduce_func:
            "reduce": alpha * temp
            "descend": temp / (1 + alpha * temp)

            where alpha is a cooling parameter

            In case of callable, no checks are implemented
            Function must have two parameters - temp and alpha and return float
            If function does not decrease temperature over time
            algorithm may not work properly

        stopping_tol:
            by default None, only stopping criterion is number of iterations

        neigh_type:
            "swap": swapping two elements in a list
            "inversion": inversing order of a slice of a list
            "insertion": inserting element into a place
        """
        # IteratingAlgorithm constructor
        super().__init__(neigh_type=neigh_type, n_iter=n_iter, verbose=verbose)
        self._stop_tool = stopping_tol
        # checks if passed reduce_func param is correct
        if not callable(reduce_func):
            assert (
                reduce_func in self._REDUCE_FUNC
            ), f"reduce_func must be one of {list(self._REDUCE_FUNC.keys())} or a function"
            reduce_func = self._REDUCE_FUNC[reduce_func]
        assert callable(reduce_func), "passed reduce_func must be callable"
        self._reduce_func: Callable = reduce_func
        # checks if alpha param is correct
        assert 0 < alpha < 1, "reduction parameter must be between 0 and 1"
        self._alpha = alpha
        # initial temp for identifying algorithm
        self._temp = self._initial_temp = temp

    def _run_iteration(self, distances: pd.DataFrame) -> None:
        # number of iteration is increased by one
        self._next_iter()
        # find radom neighbouring solution
        new_path = self._switch(distances=distances, how="random")
        # get distance of chosen random solution in vicinity
        distance = self._get_order_time(path=new_path, distances=distances)
        # distance gain
        # negative for a "good" trade; positive for a "bad" trade
        diff = distance - self._history[-1]
        # checks if temp is close to 0
        self.__check_temp()
        # calculate metropolis acceptance criterion
        # only if diff is positive, else it can lead to OverflowError
        metropolis = exp(-diff / self._temp) if diff >= 0 else 1
        # if distance is shorter - accept more optimal solution
        # if solution is worse, it can still be accepted
        # metropolis acceptance criterion is a way of not getting stuck in local minimum
        if (diff < 0) or (metropolis > random.random()):
            self._path = new_path
            self._history.append(distance)
            if self._verbose:
                print(f"step {self._i}: gain: {-diff}")
                print(f"new distance: {self._history[-1]}")

        # if new solution is rejected, algorithm goes on
        elif self._verbose:
            print(f"step {self._i}: path rejected")
            print(f"metropolis: {round(metropolis, 5)}")

        # reduce temperature
        self._temp = self._reduce_func(temp=self._temp, alpha=self._alpha)
        # in verbose mode information about switch and new temperature is printed out
        if self._verbose:
            print(f"temperature reduced to {round(self._temp, 5)}")

    def __check_temp(self) -> None:
        """
        If stopping tolerance was provided checks if temperature is close to zero
        with this absolute tolerance and raises StopAlgorithm if it is
        """
        if (self._stop_tool is not None) and isclose(
            self._temp, 0, abs_tol=self._stop_tool
        ):
            raise StopAlgorithm(iteration=self._i, distance=self._history[-1])

    def __str__(self) -> str:
        mes = super().__str__()
        mes += f"""reduction function: {self._reduce_func.__name__}\n\
            alpha: {self._alpha}\n\
            initial temperatute: {self._initial_temp}"""
        return mes
