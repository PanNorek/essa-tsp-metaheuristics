import random
from abc import ABC, abstractmethod
from typing import Union, Iterable
import pandas as pd
import numpy as np
from ..utils import (
    Inversion,
    Swap,
    Insertion,
    NeighbourhoodType,
    Result,
    get_path_distance
)


class TSPAlgorithm(ABC):
    """
    Traveling Salesman Problem (TSP) base solver

    Base, abstract class that all TSP solvers should inherit from

    Methods:
        solve - used for solving TSP problem

    Properties:
        best_path - best path found by algorithm

    The idea was that TSPAlgorithm resembles sklearn BaseEstimator interface
    where all parameters related to the algorithm itself
    and independent of the training data are passed in the constructor.
    API solve method can be use on different set of data independently of
    algorithm specifications. Most optimal path found can be accesed with
    path_ property (in sklearn attributes followed by underscore are trained from data).
    random_state and start_order parameters in solve method are an exception,
    it gives more flexibility in looking for different solutions

    The traveling salesman problem (TSP) is one of the most intensively studied
    problems in optimization.
    It is NP-hard, and the solution space scales factorially with the number of cities.
    The time complexity of the brute force approach is O(n!),
    which makes a TSP with fairly large number of cities infeasible to solve with modern computers.
    The travelling salesman problem asks the following question:
    "Given a list of cities and the distances between each pair of cities,
    what is the shortest possible route that visits each city exactly once
    and returns to the origin city?"
    It is an NP-hard problem in combinatorial optimization,
    important in theoretical computer science and operations research.

    For more information checks out:
    https://en.wikipedia.org/wiki/Travelling_salesman_problem
    https://classes.engr.oregonstate.edu/mime/fall2017/rob537/hw_samples/hw2_sample2.pdf
    """

    NAME = ""
    _NEIGHBOURHOOD_TYPES = {
        "swap": Swap,
        "inversion": Inversion,
        "insertion": Insertion,
    }

    def __init__(self,
                 neigh_type: str = "swap",
                 verbose: bool = False
                 ) -> None:
        """
        Params:
            neigh_type: str
                Type of neighbourhood used in algorithm
            verbose: bool
                If True, prints out information about algorithm progress
        """
        self._verbose = verbose
        self._path = []
        # neighbourhood class that will be used for searching
        # neighbouring solutions while solving TSP with solve method
        # src.utils.neighbourhood_type NeighbourhoodType interface that implements switch method
        neigh = self._NEIGHBOURHOOD_TYPES.get(neigh_type)
        assert (
            neigh
        ), f"neigh_type must be one of {list(self._NEIGHBOURHOOD_TYPES.keys())}"
        self._neigh_type = neigh
        # will be instanciated in solve mthod with knowledge
        # of all possible switches of cities in distance matrix
        self._neigh = None

    def solve(self,
              distances: pd.DataFrame,
              random_seed: Union[int, None] = None,
              start_order: Union[list, None] = None
              ) -> Result:
        """
        Uses specific algorithm to solve Traveling Salesman Problem

        Params:
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns
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
            distances=distances,
            random_seed=random_seed,
            start_order=start_order
        )

    @abstractmethod
    def _solve(self,
               distances: pd.DataFrame,
               random_seed: Union[int, None] = None,
               start_order: Union[list, None] = None
               ) -> Result:
        """
        Uses specific implementation of algorithm to solve Traveling Salesman Problem

        Params:
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns
            random_seed: int | None
                Seed set for all random operations inside algorithm,
                if None results won't be repeatable
            start_order: list | None
                Order from which algorithm starts solving problem,
                if None, order will be chosen randomly
        Returns:
            Result object

        Child class implementation of this method should be decorated with solve_it
        function to enhance its power.

        Decorator add solving time into the results
        and saves it directly into csv file. Moreover it enables stopping dragging out
        algorithm manually without any lose in information

        Check out src.utils.tools Result, solve_it
        """
        pass

    def _setup_start(self,
                     distances: pd.DataFrame,
                     random_seed: Union[int, None] = None,
                     start_order: Union[list, None] = None
                     ) -> list:
        """
        Sets up a starting order for an algorithm

        Params:
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns
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
        path = self._set_start_order(
            distances=distances, start_order=start_order)
        # handling starting path is a part of
        return path

    def _setup(self,
               distances: pd.DataFrame,
               random_seed: Union[int, None] = None
               ) -> None:
        """
        Sets up algorithm settings before solving the problem

        Params:
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns
            random_seed: int | None
                Seed set for all random operations inside algorithm,
                if None results won't be repeatable

        Methods checks distance matrix correctness, sets random seed
        and neighbourhood type object that is used for searching
        neighbouring solutions space
        """
        self._distances_matrix_check(distances=distances)
        self._set_random_seed(random_seed=random_seed)
        self._set_neighbourhood(distances=distances)

    def _set_neighbourhood(self, distances: pd.DataFrame) -> None:
        """
        Sets neighbourhood type object used for searching
        neighbouring solutions space

        Params:
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns

        Specified NeighbourhoodType is instantiated based on elements
        in distance matrix indices saving all possible switches.
        If algorithm has no need of using Neighbourhood type method can be
        overriden and return None, but it leaving it as it is doesn't affect
        the algorithm
        """
        self._neigh: NeighbourhoodType = self._neigh_type(
            path_length=len(distances)
        )

    def _set_start_order(self,
                         distances: pd.DataFrame,
                         start_order: Union[list, None] = None
                         ) -> list:
        """
        Returns starting solution for the problem

        Params:
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns
            start_order: list | None
                Order from which algorithm starts solving problem,
                if None, order will be chosen randomly

        If start_order is provided it checks if it's correct for a given algorithm
        and returns requested path accordingly, if not random solution is chosen.
        If algorithm requires specific checks for starting path or different
        way of chosing primary random solution it must be overriden in child class.

        For example check out src.algos.nearest_neighbour NearestNeighbour
        """
        if start_order is not None:
            self._start_order_check(start_order=start_order, distances=distances)
            return start_order
        return self._get_random_start(indices=distances.index)

    def _start_order_check(self,
                           start_order: list,
                           distances: pd.DataFrame
                           ) -> None:
        """
        Runs the series of checks to assert that starting path provided by
        user is correct and can be used in solve method

        Params:
            start_order: list | None
                Order from which algorithm starts solving problem,
                if None, order will be chosen randomly
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns

        Basic checks include:
            checking whether start_path is iterable
            checking whether start_path is of the same leghth as indices in distances matrix
            checking whether start_path alligns with distances matrix indices
            checking whether start_path has unique elements

        Specific checks are implemented in Nearestneighbour algorithm

        Check out src.algos.nearest_neighbour NearestNeighbour
        """
        assert isinstance(start_order, Iterable), 'start_order must be iterable'
        assert (
            len(start_order) == len(distances)
        ), f'Expected {len(distances)} elements, got {len(start_order)}'
        assert (
            all(index in distances.index.to_list() for index in start_order)
        ), 'elements of start_order must allign with distance matrix indices'
        assert len(set(start_order)) == len(start_order), 'elements in start_order must be unique'

    def _get_path_distance(self,
                           path: list,
                           distances: pd.DataFrame
                           ) -> Union[int, float]:
        """
        Calculates distances between cities order in the list

        Params:
            path: list
                Order of cities visited by the salesman
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns

        Wraps get_path_distance function

        Check out src.utils.tools get_path_distances
        """
        return get_path_distance(path=path, distances=distances)

    def _get_random_start(self, indices: Union[list, pd.Index]) -> list:
        """
        Chooses random path from indices as the cities

        Params:
            indices: list | pd.Index
                Cities to be visited by the salesman

        If algorithm requires different way of chosing primary random
        solution it must be overriden in child class.
        For example check out src.algos.nearest_neighbour NearestNeighbour
        which randomly selects only first cuity instead of the entire path
        """
        # indices of the cities
        if isinstance(indices, pd.Index):
            indices = list(indices)
        # return random path
        return random.sample(indices, len(indices))

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

    def _distances_matrix_check(self, distances: pd.DataFrame) -> None:
        """
        Checks if distances matrix is correct

        Params:
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns

        The only check inplemented is whether indices of distances matrix allign with
        column names. If any other check needs to be run,
        this method has to be overriden in child class
        """
        mes = "indices and columns of distances matrix should be equal"
        assert distances.index.to_list() == distances.columns.to_list(), mes

    @property
    def path_(self) -> list:
        """
        The most optimal graph's path that was found

        Path is kind of a trained attribute of the solver
        The most optimal path can be accessed at any given moment with this property
        Path is updated each time algorithm finds more optimal solution running solve method
        """
        return self._path

    def __str__(self) -> str:
        """How algorithm is represented as string in Result object and csv file"""
        return f"{self.NAME}\nNeighbourhood type: {str(self._neigh)}"

    def __repr__(self) -> str:
        """How algorithm is represented as string in Result object and csv file"""
        return str(self)
