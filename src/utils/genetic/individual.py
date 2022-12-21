import pandas as pd
from typing import Union
from ..neighbourhood_type import NeighbourhoodType


class Individual:
    PATH = "path"
    DISTANCE = "distance"
    INDIVIDUAL = "individual"
    FITNESS = "fitness"

    def __init__(self, path: list, distance: Union[int, float]) -> None:
        self.path = path
        self.distance = distance

    @property
    def fitness(self) -> float:
        return 1 / self.distance

    def mutate(self, neigh_type: NeighbourhoodType) -> None:
        self.path = neigh_type.switch(path=self.path, how='random')

    def __assert_type(self, object_) -> None:
        assert isinstance(object_, Individual), f"Cannot compare with {type(object_)} type"

    def __gt__(self, object) -> bool:
        self.__assert_type(object)
        return self.fitness > object.fitness

    def __str__(self) -> str:
        return f"Individual(path: {self.path[0]}...{self.path[-1]}, distance: {self.distance:.2f})"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.path)

    @property
    def __dict__(self) -> dict:
        return {
            self.INDIVIDUAL: self,
            self.PATH: self.path,
            self.DISTANCE: self.distance,
            self.FITNESS: self.fitness,
        }
