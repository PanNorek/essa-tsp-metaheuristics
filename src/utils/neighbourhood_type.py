from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import random
from typing import List
import pandas as pd
from .distance import get_path_distance
from .queue_list import Queue


class NeighbourhoodType(ABC):
    """Abstract class for mutable"""
    NAME = ''
    _SWITCH_OPTIONS = ['best', 'random']

    def __init__(self, path_length: int) -> None:
        self._switches = self._get_all_switches(length=path_length)

    @property
    def last_switch(self):
        return self._last_switch

    @abstractproperty
    def last_switch_comment(self) -> str:
        pass

    def switch(self,
               path: list,
               distances: pd.DataFrame = None,
               how: str = 'best',
               exclude: Queue | None = None
               ) -> list:
        """Returns copy of the current path with swapped indices"""
        if distances is None and how == 'best':
            raise ValueError('Cannot find best switch without distances matrix')
        path = path[:]
        assert how in self._SWITCH_OPTIONS, f'how must be one of {self._SWITCH_OPTIONS}'
        if how == 'best':
            switch = self._find_best_switch(path=path,
                                            distances=distances,
                                            exclude=exclude)
        elif how == 'random':
            rnd_idx = random.randint(0, len(self._switches) - 1)
            switch = self._switches[rnd_idx]

        self._last_switch = switch
        self._last_path = path
        new_path = self._switch(path=path,
                                index_1=switch[0],
                                index_2=switch[1])
        return new_path

    def _find_best_switch(self,
                          path: list,
                          distances: pd.DataFrame,
                          exclude: Queue | None = None
                          ) -> tuple:
        assert len(path) == len(distances), 'path and distances df must have the same length'
        legal_switches = self._exclude_switches(exclude=exclude)
        new_paths = [
               (self._switch(path=path, index_1=switch[0], index_2=switch[1]), switch)
               for switch in legal_switches
        ]
        new_paths.sort(key=lambda x: get_path_distance(path=x[0], distances=distances))
        return new_paths[0][1]

    def _exclude_switches(self, exclude: Queue | None = None) -> List[tuple]:
        return list(set(self._switches) - set(exclude))

    @abstractmethod
    def _get_all_switches(self, length: int) -> List[tuple]:
        """Returns all possible swaps of indices"""
        pass

    @abstractmethod
    def _switch(self, path: list, index_1: int, index_2: int) -> list:
        pass

    def __str__(self) -> str:
        return self.NAME

    def __repr__(self) -> str:
        return str(self)


class Swap(NeighbourhoodType):
    NAME = 'Swap'

    def _switch(self, path: list, index_1: int, index_2: int) -> list:
        path = path[:]
        path[index_1], path[index_2] = path[index_2], path[index_1]
        return path

    def _get_all_switches(self, length: int) -> List[tuple]:
        """Returns all possible swaps of indices"""
        # unique combination
        swaps = [
            (x, y)
            for x in range(length)
            for y in range(length)
            if (x != y) and (y > x)
        ]
        return swaps

    @property
    def last_switch_comment(self) -> str:
        if not hasattr(self, '_last_path') or not hasattr(self, '_last_switch'):
            return ''
        swapped_1, swapped_2 = [self._last_path[index] for index in self._last_switch]
        return f"{swapped_1} swapped with {swapped_2}"


class Insertion(NeighbourhoodType):
    NAME = 'Insertion'

    def _switch(self, path: list, index_1: int, index_2: int) -> list:
        path = path[:]
        path.insert(index_1, path.pop(index_2))
        return path

    def _get_all_switches(self, length: int) -> List[tuple]:
        """Returns all possible swaps of indices"""
        swaps = [
            (x, y)
            for x in range(length)
            for y in range(length)
            if (x != y)
        ]
        return swaps

    @property
    def last_switch_comment(self) -> str:
        if not hasattr(self, '_last_path') or not hasattr(self, '_last_switch'):
            return ''
        element = self._last_path[self._last_switch[1]]
        index = self._last_switch[0]
        return f"{element} at index {self._last_switch[1]} inserted at index {index}"


class Inversion(NeighbourhoodType):
    """Inversion mutation"""
    NAME = 'Inversion'

    def __init__(self,
                 path_length: int,
                 window_length: int | None = None
                 ) -> None:
        if window_length is None:
            window_length = path_length//10
        window_length = max(window_length, 3)
        self._window_length = window_length
        super().__init__(path_length=path_length)

    def _switch(self, path: list, index_1: int, index_2: int) -> list:
        path = path[:]
        path[index_1: index_2] = path[index_1: index_2][::-1]
        return path

    def _get_all_switches(self, length: int) -> List[tuple]:
        """Returns all possible swaps of indices"""
        # unique combination
        swaps = [
            (x, y)
            for x in range(length)
            for y in range(length)
            if (x != y) and (y > x) and (y - x == self._window_length)
        ]
        return swaps

    @property
    def last_switch_comment(self) -> str:
        if not hasattr(self, '_last_path') or not hasattr(self, '_last_switch'):
            return ''
        elements = [self._last_path[index] for index in range(*self._last_switch)]
        return f"Elements {elements} at indices {self._last_switch} inversed"
