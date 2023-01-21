import random
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from ..utils import (
    Result,
    get_order_cost,
    path_check,
    distances_matrix_check,
    get_random_path,
)


class PFSPAlgorithm(ABC):
    """
    Permutation Flowshop Scheduling Problem (PFSP) base solver

    Base, abstract class that all PFSP solvers should inherit from

    Methods:
        solve - used for solving PFSP problem

    Attributes:
        path_ - best order found by algorithm

    The idea was that PFSPAlgorithm resembles sklearn BaseEstimator interface
    where all parameters related to the algorithm itself
    and independent of the training data are passed in the constructor.
    API solve method can be use on different set of data independently of
    algorithm specifications. Optimal order found can be accesed with
    path_ property (in sklearn attributes followed by underscore are trained from data).
    random_state and start_order parameters in solve method are an exception,
    it gives more flexibility in looking for different solutions

    The Permutation Flowshop Scheduling Problem (PFSP) is a type of scheduling problem that involves scheduling a set of jobs on a set of machines in a specific order, such that the overall completion time is minimized. In a PFSP, the jobs must be processed in a specific order, and cannot be reordered or split up. The objective is to find the optimal order in which to process the jobs on the machines in order to minimize the overall completion time. The PFSP is a NP-hard problem, meaning that it is computationally difficult to find an exact solution in a reasonable amount of time for large problem instances.

    The time complexity of a brute force algorithm for the PFSP would be O(n!), where n is the number of jobs.


    For more information check out:
    https://en.wikipedia.org/wiki/Flow-shop_scheduling
    """

    NAME = ""

    def __init__(self, verbose: bool = False) -> None:
        """
        Params:
            verbose: bool
                If True, prints out information about algorithm progress
        """
        self._verbose = verbose
        # path before solving is an empty list
        self._path = []

    def solve(
        self,
        distances: pd.DataFrame,
        random_seed: Union[int, None] = None,
        start_order: Union[list, None] = None,
    ) -> Result:
        """
        Uses specific algorithm to solve Permutation Flowshop Scheduling Problem

        Params:
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order
            random_seed: int | None
                Seed set for all random operations inside algorithm,
                if None results won't be repeatable
            start_order: list | None
                Order from which algorithm starts solving problem,
                if None, order will be chosen randomly
        Returns:
            Result object

        Method is a wrapper around _solve method - specific implementation of child class

        Check out src.utils.tools Result
        """
        # all classes inherited from Algorithm must have protected _solve method
        # this method includes all specific operations going on inside
        return self._solve(
            distances=distances, random_seed=random_seed, start_order=start_order
        )

    @abstractmethod
    def _solve(
        self,
        distances: pd.DataFrame,
        random_seed: Union[int, None] = None,
        start_order: Union[list, None] = None,
    ) -> Result:
        """
        Uses specific implementation of algorithm to solve Permutation Flowshop Scheduling Problem

        Params:
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order
            random_seed: int | None
                Seed set for all random operations inside algorithm,
                if None results won't be repeatable
            start_order: list | None
                Order from which algorithm starts solving problem,
                if None, order will be chosen randomly
        Returns:
            Result object

        Developer note:

            Child class implementation of this method should be decorated with solve_it
            function to enhance its power.

            Decorator add solving time into the results
            and saves it directly into csv file. Moreover it enables stopping dragging out
            algorithm manually without any lose in information

        Check out src.utils.tools Result, solve_it
        """
        pass

    def _setup_start(
        self,
        distances: pd.DataFrame,
        random_seed: Union[int, None] = None,
        start_order: Union[list, None] = None,
    ) -> list:
        """
        Sets up a starting order for an algorithm

        Params:
            distances: pd.DataFrame
               Matrix of set of jobs scheduled on a set of machines in a specific order
            random_seed: int | None
                Seed set for all random operations inside algorithm,
                if None results won't be repeatable
            start_order: list | None
                Order from which algorithm starts solving problem,
                if None, order will be chosen randomly
        Returns:
            list representing starting path

        Methods checks distance matrix correctness, sets random seed
        and neighbourhood type object that is used for searching
        neighbouring solutions space.

        Child classes should use this method inside _solve method
        or reimplement it if need be.
        """
        self._setup(distances=distances, random_seed=random_seed)
        # staring path or its part if method is overriden as of ex. Nearest Neighbour
        path = self._set_start_order(distances=distances, start_order=start_order)
        # handling starting path is a part of
        return path

    def _setup(
        self, distances: pd.DataFrame, random_seed: Union[int, None] = None
    ) -> None:
        """
        Sets up algorithm settings before solving the problem

        Params:
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order
            random_seed: int | None
                Seed set for all random operations inside algorithm,
                if None results won't be repeatable

        Methods checks distance matrix correctness and sets random seed
        """
        self._distances_matrix_check(distances=distances)
        self._set_random_seed(random_seed=random_seed)

    def _set_start_order(
        self, distances: pd.DataFrame, start_order: Union[list, None] = None
    ) -> list:
        """
        Returns starting solution for the problem

        Params:
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order
            start_order: list | None
                Order from which algorithm starts solving problem,
                if None, order will be chosen randomly

        If start_order is provided it checks if it's correct for a given algorithm
        and returns requested order accordingly, if not random solution is chosen.
        If algorithm requires specific checks for starting order or different
        way of chosing primary random solution it must be overriden in child class.

        For example check out

        src.algos.nearest_neighbour NearestNeighbour
        """
        if start_order is not None:
            self._start_order_check(start_order=start_order, distances=distances)
            return start_order
        return self._get_random_start(indices=distances.index)

    def _start_order_check(self, start_order: list, distances: pd.DataFrame) -> None:
        """
        Runs the series of checks to assert that starting path provided by
        user is correct and can be used in solve method

        Params:
            start_order: list
                Order from which algorithm starts solving problem
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order

        Basic checks include:
            # checking whether start_path is iterable #TODO: add comments
            # checking whether start_path is of the same leghth as indices in distances matrix
            # checking whether start_path alligns with distances matrix indices
            # checking whether start_path has unique elements

        This method wraps path_check function from src.utils.tools

        Specific checks are implemented in PFSP_NearestNeighbor algorithm

        Check out:

        src.algos.pfsp_nn PFSP_NearestNeighbor
        """
        path_check(path=start_order, distances=distances)
        assert isinstance(start_order, list), "start_order must be a list"

    def _get_order_time(self, path: list, distances: pd.DataFrame) -> Union[int, float]:
        """
        Calculates total time based on given order and distances matrix

        Params:
            path: list
                Order of tasks to be scheduled
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order
        Wraps get_order_cost function

        Check out:

        src.utils.tools get_order_cost
        """
        return get_order_cost(order=path, cost_matrix=distances)

    def _get_random_start(self, indices: Union[list, pd.Index]) -> list:
        """
        Chooses random path from indices as the cities

        Params:
            indices: list | pd.Index
                List of indices to choose from

        If algorithm requires different way of chosing primary random
        solution it must be overriden in child class.
        """
        return get_random_path(indices=indices)

    def _set_random_seed(self, random_seed: Union[int, None] = None) -> None:
        """
        Sets random seed for the algorithm so as to make all
        results repeatable

        Params:
            random_seed: int | None
                Seed set for all random operations inside algorithm,
                if None results won't be repeatable

        Note: method sets random seed for numpy as well!
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

    # def _distances_matrix_check(self, distances: pd.DataFrame) -> None:
    #     """
    #     Checks if distances matrix is correct

    #     Params:
    #         distances: pd.DataFrame
    #             Matrix of distances between cities,
    #             cities numbers or id names as indices and columns

    #     The only check inplemented is whether indices of distances matrix allign with
    #     column names. Wraps distances_matrix_check function.
    #     If any other check needs to be run, this method has to be overriden in child class

    #     Check out:

    #     src.utils.tools distances_matrix_check
    #     """
    #     # distances_matrix_check(distances=distances)
    #     pass

    @property
    def path_(self) -> list:
        """
        The optimal graph's order that was found

        Path is kind of a trained attribute of the solver
        The optimal path can be accessed at any given moment with this property
        Path is updated each time algorithm finds more optimal solution running solve method
        """
        return self._path

    def __str__(self) -> str:
        """How algorithm is represented as string in Result object and csv file"""
        return f"{self.NAME}\n"

    def __repr__(self) -> str:
        """How algorithm is represented as string in Result object and csv file"""
        return str(self)
