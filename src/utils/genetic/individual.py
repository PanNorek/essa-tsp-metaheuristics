import pandas as pd
from typing import Union, List
import numpy as np

class Individual:
    PATH = 'path'
    DISTANCE = 'distance'
    INDIVIDUAL = 'individual'
    FITNESS = 'fitness'

    def __init__(self, path: list, distance: Union[int, float]) -> None:
        self.path = path
        self.distance = distance

    @property
    def fitness(self) -> float:
        return 1 / self.distance

    def mutate(self, mutation_rate: int = 0) -> None:
        q = 1 - mutation_rate  # 1 - probability of mutation
        n = len(self.path)
        for swapped in range(n):
            if np.random.choice((True, False), p=[mutation_rate, q]):
                swap_with = np.random.choice(range(n))
                self.path[swapped], self.path[swap_with] = self.path[swap_with], self.path[swapped]

    def to_df(self):
        dic = {self.INDIVIDUAL: [self],
               self.PATH: [self.path],
               self.DISTANCE: [self.distance],
               self.FITNESS: [self.fitness]}
        df = pd.DataFrame(dic).sort_values(self.FITNESS, ascending=False)
        return df

    def __assert_type(self, object_) -> None:
        assert isinstance(object_, Individual), f"Cannot compare with {type(object_)} type"

    def __gt__(self, object) -> bool:
        self.__assert_type(object)
        return self.fitness > object.fitness

    def __str__(self) -> str:
        return f"Individual(path: {self.path[0]}...{self.path[-1]}, distance: {self.distance})"

    def __repr__(self) -> str:
        return str(self)
