from typing import Union, Iterable
import pandas as pd
from .algorithm import TSPAlgorithm
from ..utils import (
    Inversion,
    Swap,
    Insertion,
    NeighbourhoodType,
)


class TSPHeuristicAlgorithm(TSPAlgorithm):
    """
    Solver for heuristic approach to Traveling Salesman Problem (TSP)

    Methods:
        solve - used for solving TSP problem

    Attributes:
        path_ - best path found by algorithm

    Heuristic is a technique designed for solving a problem more quickly
    when classic methods are too slow for finding an approximate solution,
    or when classic methods fail to find any exact solution.
    This is achieved by trading optimality, completeness, accuracy, or precision for speed.
    In a way, it can be considered a shortcut

    In TSP checking every path is taking extreme amount of time,
    heuristic algorithms narrow down the search space only to adjecent solutions

    Interface wraps functionality of NeighbourhoodType interface
    facilitating further development. Neighbourhood type setup
    proper for provided distance matrix is done in the background

    For more information check out:
    https://en.wikipedia.org/wiki/Travelling_salesman_problem
    https://en.wikipedia.org/wiki/Heuristic_(computer_science)

    src.utils.neighbourhood_type NeighbourhoodType
    """

    _NEIGHBOURHOOD_TYPES = {
        "swap": Swap,
        "inversion": Inversion,
        "insertion": Insertion,
    }

    def __init__(self, neigh_type: str = "swap", verbose: bool = False) -> None:
        """
        Params:
            neigh_type: str
                Type of neighbourhood used in algorithm
            verbose: bool
                If True, prints out information about algorithm progress

        neigh_type:
            "swap": swapping two elements in a list
            "inversion": inversing order of a slice of a list
            "insertion": inserting element into a place
        """

        super().__init__(verbose=verbose)

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

    def _setup(
        self, distances: pd.DataFrame, random_seed: Union[int, None] = None
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
        super()._setup(distances=distances, random_seed=random_seed)
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
        self._neigh: NeighbourhoodType = self._neigh_type(path_length=len(distances))

    def _switch(
        self,
        distances: Union[pd.DataFrame, None],
        how: str = "best",
        exclude: Union[Iterable, None] = None,
    ) -> list:
        """
        Uses switch specific to the neighbourhood type on the current solution
        and returns the new adjacent solution

        Params:
            distances: pd.DataFrame | None
                Matrix of distances between cities,
                cities numbers or id names as indices and columns
            how: str
                The way adjacent solution is chosen
            exclude: list | None
                List of forbidden switches
        Returns:
            a copy of a current solution with modified order
            representing a new adjacent solution

        distances:
            distances matrix can be None if how="random"
            there is no need for calculating solutions distances then
        how:
            "best" - default option
                Every possible solution in vicinity is checked and optimal is returned
            "random" - random solution from vicinity is returned

        exclude:
            list of forbidden switches, by default None
            Provide a list of tuples of indices ex. [(1,4), (2,6)]
            Used in TabuSearch as a way to escape local mininum

            Check out:

            src.algos.tabu_search TabuSearch

        Wraps NeighbourhoodType switch method

        Check out:

        src.utils.neighbourhood_type NeighbourhoodType
        """
        return self._neigh.switch(
            path=self._path, distances=distances, how=how, exclude=exclude
        )

    @property
    def _last_switch(self) -> tuple:
        """Tuple representing the latest switch"""
        return self._neigh.last_switch

    @property
    def _last_switch_comment(self) -> str:
        """Explicit comment on the latest switch"""
        return self._neigh.last_switch_comment

    def __str__(self) -> str:
        mes = super().__str__()
        mes += f"Neighbourhood type: {str(self._neigh)}"
        return mes
