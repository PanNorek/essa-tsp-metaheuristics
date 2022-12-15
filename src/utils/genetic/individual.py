import pandas as pd
from typing import Union, List
import numpy as np
from .mutable import Mutable


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

    def mutate(self, mutation_type: Mutable, mutation_rate: float) -> None:
        """Mutate individual

        Args:
            mutation_rate (int, optional): _description_. Defaults to 0.
            neigh_type (str, optional): _description_. Defaults to "simple".
        """

        mutation_type.mutate(self, mutation_rate)

    def to_df(self):
        dic = {
            self.INDIVIDUAL: [self],
            self.PATH: [self.path],
            self.DISTANCE: [self.distance],
            self.FITNESS: [self.fitness],
        }
        return pd.DataFrame(dic).sort_values(self.FITNESS, ascending=False)

    def __assert_type(self, object_) -> None:
        assert isinstance(object_, Individual), f"Cannot compare with {type(object_)} type"

    def __gt__(self, object) -> bool:
        self.__assert_type(object)
        return self.fitness > object.fitness

    def __str__(self) -> str:
        return f"Individual(path: {self.path[0]}...{self.path[-1]}, distance: {self.distance:.2f})"

    def __repr__(self) -> str:
        return str(self)
