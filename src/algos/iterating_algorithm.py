from abc import abstractmethod
from typing import Union
import pandas as pd
from .heuristic_algorithm import PFSPHeuristicAlgorithm
from ..utils import StopAlgorithm, Result, solve_it


class IteratingAlgorithm(PFSPHeuristicAlgorithm):
    """
    Solver for iterative approach to heuristic algorithms
    for Permutation Flowshop Scheduling Problem (PFSP)

    Methods:
        solve - used for solving PFSP problem

    Attributes:
        path_ - best order found by algorithm
        history - list of accepted solutions during iteration process

    Interface is designed for heuristic algorithms, it inherits from
    PFSPHeuristicAlgorithm providing functionality for searching
    neighbouring solutions space.

    Developer note:

        The assumptions is that each step in iterative process is the same.

        Specific logic must be implemented in child classes in _run_iteration
        method.

        Specific default number of iteration can be set in
        class attribute DEFAULT_ITERS.

        Algorithm can be stopped with StopAlgorithm Exception in _run_iteration
        method. Algorithm keeps track of the number of current iteration.
        In can be accessed with _i attribute.

        Accepted solutions can be accessed with history attributes.
        To check the latest solution get the last element of the list
        ex. self._history[-1] or the best solution min(self._history)

        New solution and distance history record must be set inside
        _run_iteration method

    Check out:

    src.utils.neighbourhood_type NeighbourhoodType

    src.utils.heuristic_algorithm PFSPHeuristicAlgorithm

    src.utils.tools StopAlgorithm
    """

    DEFAULT_ITERS = 30

    def __init__(
        self,
        neigh_type: str = "swap",
        n_iter: int = DEFAULT_ITERS,
        verbose: bool = False,
    ) -> None:
        """
        Params:
            neigh_type: str
                Type of neighbourhood used in algorithm
            n_iter: int
                Number of iterations to be run
            verbose: bool
                If True, prints out information about algorithm progress

        neigh_type:
            "swap": swapping two elements in a list
            "inversion": inversing order of a slice of a list
            "insertion": inserting element into a place
        """
        super().__init__(neigh_type=neigh_type, verbose=verbose)
        self._n_iter = n_iter
        # current iteration
        self._i = 0
        self._history = []

    @solve_it
    def _solve(
        self,
        distances: pd.DataFrame,
        random_seed: Union[int, None] = None,
        start_order: Union[list, None] = None,
    ) -> Result:
        self._path = self._setup_start(
            distances=distances, random_seed=random_seed, start_order=start_order
        )
        # distance of the path at the beginning
        distance = self._get_order_time(path=self._path, distances=distances)
        # list of distances at i iteration
        self._history = [distance]

        if self._verbose:
            print(f"--- {self.NAME} ---")
            print(f"step {self._i}: distance: {self._history[-1]}")

        self._iterate(distances=distances)

        # return result object
        return Result(
            algorithm=self,
            path=self._path,
            distance=self._history[-1],
            distance_history=self._history,
        )

    def _iterate(self, distances: pd.DataFrame) -> None:
        """
        Uses specific algorithm to solve Traveling Salesman Problem

        Params:
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order

        Runs a loop with operations specific to an algorithm

        Break conditions:
            - set number of iteration is completed
            - algorithm was stopped manually
            - StopAlgorithm exception was raised
        """
        for _ in range(self._n_iter):
            try:
                self._run_iteration(distances=distances)
            except StopAlgorithm as exc:
                exc.iteration = self._i
                exc.distance = self.history[-1]
                if self._verbose:
                    print(exc.info)
                break

    @abstractmethod
    def _run_iteration(self, distances: pd.DataFrame) -> None:
        """Runs operations in a single interation

        Params:
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order

        It saves all information in attributes
        """
        pass

    @property
    def history(self) -> list:
        """list of accepted solutions during iteration process"""
        return self._history

    def _next_iter(self) -> None:
        """Sets up algorithm for the next iteration"""
        # start new iteration
        self._i += 1

    def _reset_iter(self) -> None:
        """Resets algorithm before next run"""
        self._i = 0
        self._history = []

    def _setup_start(
        self,
        distances: pd.DataFrame,
        random_seed: Union[int, None] = None,
        start_order: Union[list, None] = None,
    ) -> list:
        # resets algorithm
        self._reset_iter()
        return super()._setup_start(
            distances=distances, random_seed=random_seed, start_order=start_order
        )

    def __str__(self) -> str:
        mes = super().__str__()
        mes += f"""\nn_iter: {self._n_iter}\n"""
        return mes
