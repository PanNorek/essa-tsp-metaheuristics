from abc import ABC, abstractmethod, abstractproperty
import random
from typing import List, Union
import pandas as pd
from .tools import get_order_cost, path_check, StopAlgorithm


class NeighbourhoodType(ABC):
    """
    NeighbourhoodType interface

    Base, abstract class that all NieghbourhoodType classes should inherit from

    Methods:
        switch - switches elements of the list and
            returns a modified one

    Properties:
        last_switch - indices that were switched recently
        last_switch_comment - explicit comment on the recent switch

    Neighbourhood is a crucial part of TSP algorithms. With goal of finding
    more optimal path, algorithm needs to search the vicinity of the current solution.

    For move to be considered a neighbourhood it must fullfill these requirements:
        - the number of adjacent solutions must be small
        - possibility of reaching every solution from initial solution
        - solutions in vicinity must be similar

    The TSPAlgotithm uses different kind of NeighbourhoodType to search neighbouring
    solutions space in iterative manner to reach optimal solution

    Algorithm keeps all possible switches in an attribute nad implements
    random switch and best switch (one that gives optimal adjacent solution)
    Best switch is a default option, but it's more time consuming. Every possible
    solution in vicinity is checked and optimal is returned
    For reference: SimulatedAnnealing uses random switch at each step,
    HillClimbing always looks for the best solution in the neighbourhood

    Check out:

    src.algos.simulated_annealing SimulatedAnnealing

    src.algos.hill_climbing HillClimbing
    """

    NAME = ""
    _SWITCH_OPTIONS = ["best", "random"]

    def __init__(self, path_length: int) -> None:
        """
        Params:
            path_length: int
                Lenght of ther path representing the order of cities

        Interface assumes that switch method always takes a list.
        For simplicity, indices are switched, not actual values.
        That's why constructor takes path length instead of the names of cities
        """
        # all possible switches are constructed and kept in an attribute
        # for quick access
        self._switches = self._get_all_switches(length=path_length)
        self._last_switch = ()
        self._last_path = []

    @property
    def last_switch(self) -> tuple:
        """Tuple representing the latest switch"""
        return self._last_switch

    @abstractproperty
    def last_switch_comment(self) -> str:
        """Explicit comment on the latest switch"""
        pass

    def switch(
        self,
        path: list,
        distances: Union[pd.DataFrame, None] = None,
        how: str = "best",
        exclude: Union[list, None] = None,
    ) -> list:
        """
        Uses switch specific to the neighbourhood type and returns
        the new adjacent solution

        Params:
            path: list
                Order of cities visited by the salesman
            distances: pd.DataFrame | None
                Matrix of set of jobs scheduled on a set of machines in a specific order
            how: str
                The way adjacent solution is chosen
            exclude: list | None
                List of forbidden switches
        Returns:
            a copy of a list with modified order
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
        """
        # for searching for the optimal solution in vicinity
        # distances matrix is needed
        if distances is None and how == "best":
            raise ValueError("Cannot find best switch without distances matrix")
        # copy of the initial path
        path = path[:]
        # check if correct option was passed
        assert how in self._SWITCH_OPTIONS, f"how must be one of {self._SWITCH_OPTIONS}"
        if how == "best":
            # some neighbourhood types require specific way of
            # finding optimal solution in the vicinity
            switch = self._find_best_switch(
                path=path, distances=distances, exclude=exclude
            )
        elif how == "random":
            # in case of random solution, process is the same for all neighbourhood types
            rnd_idx = random.randint(0, len(self._switches) - 1)
            switch = self._switches[rnd_idx]
        # initial state is stored in attributes for easy access and information
        # about modification from outside of the class (in algorithm)
        # SwitchingAgorithm implements wrapper for easy access to these attributes
        self._last_switch = switch
        self._last_path = path
        # each neighbourhood type has its own implementation of _switch
        # current ones always take path and tuple of indices as parameters
        new_path = self._switch(path=path, switch=switch)
        # new adjacent solution is returned
        return new_path

    def _find_best_switch(
        self, path: list, distances: pd.DataFrame, exclude: Union[list, None] = None
    ) -> tuple:
        """
        Uses specific logic to find the best adjacent solution

        Params:
            path: list
                Order of cities visited by the salesman
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order
            exclude: list | None
                List of forbidden switches
        Returns:
            a tuple with indices representing a switch for best
            adjacent solution

        exclude:
            list of forbidden switches, by default None
            Provide a list of tuples of indices ex. [(1,4), (2,6)]
            Used in TabuSearch as a way to escape local mininum

            Check out:

            src.algos.tabu_search TabuSearch
        """
        # checks if path is correct
        self._order_check(path=path, distances=distances)
        # exclude all forbidden switches
        legal_switches = self._exclude_switches(exclude=exclude)

        if not legal_switches:
            raise StopAlgorithm(message="No legal neighbouring solution available")

        # list of tuples of all solutions in vicinity and their switches
        new_paths = [
            (self._switch(path=path, switch=switch), switch)
            for switch in legal_switches
        ]
        # sort it by distance
        new_paths.sort(key=lambda x: get_order_cost(order=x[0], cost_matrix=distances))
        # return switch that results in optimal adjacent solution
        return new_paths[0][1]

    def _order_check(self, path: list, distances: pd.DataFrame) -> None:
        path_check(path=path, distances=distances)

    def _exclude_switches(self, exclude: Union[list, None] = None) -> List[tuple]:
        """
        Excludes forbidden switches and returns list of all legal switches

        Params:
            exclude: list | None
                List of forbidden switches
        Returns:
            a list of tuples with indices representing all
            legal switches

        exclude:
            list of forbidden switches, by default None
            Provide a list of tuples of indices ex. [(1,4), (2,6)]
            Used in TabuSearch as a way to escape local mininum

            Check out:

            src.algos.tabu_search TabuSearch
        """
        return (
            list(set(self._switches) - set(exclude))
            if exclude
            else list(set(self._switches))
        )

    @abstractmethod
    def _get_all_switches(self, length: int) -> List[tuple]:
        """
        Returns all possible switches of indices for a neighbourhood type

        Params:
            length: int
                length of the list representing an order of cities
        Returns:
            a list of tuples with indices representing all
            possible switches
        """
        pass

    @abstractmethod
    def _switch(self, path: list, switch: tuple) -> list:
        """
        Apply a modification to the path transforming it into
        an adjacent solution

        Params:
            path: list
                Order of cities visited by the salesman
            switch: tuple
                Indices used for switch

        Returns:
            a list representing a new path, which is a solution
            from vicinity of the initial path
        """
        pass

    def __str__(self) -> str:
        """String representation of the object"""
        return self.NAME

    def __repr__(self) -> str:
        """String representation of the object"""
        return str(self)


class Swap(NeighbourhoodType):
    """
    Swap is a type of neighbourhood used in TSP

    Methods:
        switch - switches elements of the list and
            returns a modified one

    Properties:
        last_switch - indices that were switched recently
        last_switch_comment - explicit comment on the recent switch

    Implements:
        NeighbourhoodType - base class for all neighbourhoods

    For Swap adjacent solution is constructed as follows:
        initial path: [1, 5, 6, 9, 0, 2]
        switch indices: (1, 3)
        new solution: [1, 9, 6, 5, 0, 2]
        Elements at indices 1 and 3 were swapped

    Check out:

    src.utils.neighbourhood_type NeighbourhoodType interface
    """

    NAME = "Swap"

    def _switch(self, path: list, switch: tuple) -> list:
        path = path[:]
        index_1, index_2 = switch
        # swaps two elements
        path[index_1], path[index_2] = path[index_2], path[index_1]
        return path

    def _get_all_switches(self, length: int) -> List[tuple]:
        # unique combinations
        swaps = [
            (x, y) for x in range(length) for y in range(length) if (x != y) and (y > x)
        ]
        return swaps

    @property
    def last_switch_comment(self) -> str:
        if not hasattr(self, "_last_path") or not hasattr(self, "_last_switch"):
            return ""
        swapped_1, swapped_2 = [self._last_path[index] for index in self._last_switch]
        return f"{swapped_1} swapped with {swapped_2}"


class Insertion(NeighbourhoodType):
    """
    Insertion is a type of neighbourhood used in TSP

    Methods:
        switch - switches elements of the list and
            returns a modified one

    Properties:
        last_switch - indices that were switched recently
        last_switch_comment - explicit comment on the recent switch

    Implements:
        NeighbourhoodType - base class for all neighbourhoods

    For Insertion adjacent solution is constructed as follows:
        initial path: [1, 5, 6, 9, 0, 2]
        switch indices: (1, 3)
        new solution: [1, 6, 9, 5, 0, 2]
        Elements at index 1 was inserted into index 3

    The slowest one of neighbourhood types while
    searching for the best adjacent solution.
    The number of switches is significantly larger.

    Check out:

    src.utils.neighbourhood_type NeighbourhoodType interface
    """

    NAME = "Insertion"

    def _switch(self, path: list, switch: tuple) -> list:
        path = path[:]
        index_1, index_2 = switch
        # inserts element at index index_1 into index_2
        path.insert(index_2, path.pop(index_1))
        return path

    def _get_all_switches(self, length: int) -> List[tuple]:
        swaps = [(x, y) for x in range(length) for y in range(length) if (x != y)]
        return swaps

    def _exclude_switches(self, exclude: Union[list, None] = None) -> List[tuple]:
        # include reverse operation to prevent algorithm from accepting recent solution again
        return (
            list(set(self._switches) - set(
                list(exclude) + [self._last_switch[::-1]]
            ))
            if exclude
            else list(set(self._switches))
        )

    @property
    def last_switch_comment(self) -> str:
        if not hasattr(self, "_last_path") or not hasattr(self, "_last_switch"):
            return ""
        element = self._last_path[self._last_switch[0]]
        index = self._last_switch[1]
        return f"{element} at index {self._last_switch[0]} inserted at index {index}"


class Inversion(NeighbourhoodType):
    """
    Inversion is a type of neighbourhood used in TSP

    Methods:
        switch - switches elements of the list and
            returns a modified one

    Properties:
        last_switch - indices that were switched recently
        last_switch_comment - explicit comment on the recent switch

    Implements:
        NeighbourhoodType - base class for all neighbourhoods

    For Inversion adjacent solution is constructed as follows:
        initial path: [1, 5, 6, 9, 0, 2]
        switch indices: (1, 3)
        new solution: [1, 9, 6, 5, 0, 2]
        Slice from index 1 to index 3 was inversed

    The quickest one of neighbourhood types while
    searching for the best adjacent solution.
    The number of switches is significantly smaller.

    Check out:

    src.utils.neighbourhood_type NeighbourhoodType interface
    """

    NAME = "Inversion"

    def __init__(
        self, path_length: int, window_length: Union[int, None] = None
    ) -> None:
        """
        Params:
            path_length: int
                Lenght of ther path representing the order of cities
            window_lenth: int | None
                Length of a slice to be inversed

        window_length:
            Gives a flexibility of chosing a desired window_length
            that is being inversed in switch method
            If not passed, optimal length will be chosen

        Interface assumes that switch method always takes a list.
        For simplicity, indices are switched, not actual values.
        That's why constructor takes path length instead of the names of cities
        """
        # sets window_length
        # this option is not used with Algorithm, to reduce complexity
        if window_length is None:
            window_length = path_length // 10
        window_length = max(window_length, 3)
        assert (
            window_length < path_length
        ), "window length cannot be greater than path_length"
        self._window_length = window_length
        super().__init__(path_length=path_length)

    def _switch(self, path: list, switch: tuple) -> list:
        path = path[:]
        index_1, index_2 = switch
        # inversion of a specified slice
        path[index_1:index_2] = path[index_1:index_2][::-1]
        return path

    def _get_all_switches(self, length: int) -> List[tuple]:
        # unique combinations
        swaps = [
            (x, y)
            for x in range(length)
            for y in range(length)
            if (x != y) and (y > x) and (y - x == self._window_length)
        ]
        return swaps

    @property
    def last_switch_comment(self) -> str:
        if not hasattr(self, "_last_path") or not hasattr(self, "_last_switch"):
            return ""
        elements = [self._last_path[index] for index in range(*self._last_switch)]
        return f"Elements {elements} at indices {self._last_switch} inversed"


class NEHInsertion(Insertion):

    def _get_all_switches(self, length: Union[int, None]) -> list[tuple]:
        if length is None:
            return ()
        swaps = [(length-1, x) for x in range(length)]
        return swaps

    def switch(
        self,
        path: list,
        distances: Union[pd.DataFrame, None] = None,
        how: str = "best",
        exclude: Union[list, None] = None,
    ) -> list:
        self._switches = self._get_all_switches(length=len(path))
        return super().switch(
            path=path,
            distances=distances,
            how=how,
            exclude=exclude
        )

    def _order_check(self, path: list, distances: pd.DataFrame) -> None:
        pass
