import pandas as pd
from typing import Union, List


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
        return 1/self.distance

    def to_df(self):
        dic = {self.INDIVIDUAL: [self],
               self.PATH: [self.path],
               self.DISTANCE: [self.distance],
               self.FITNESS: [self.fitness]}
        df = pd.DataFrame(dic)
        df = df.sort_values(self.FITNESS, ascending=False)
        return df

    def mate(self, object):
        self.__assert_type(object)
        pass

    def __assert_type(self, object) -> None:
        assert isinstance(object, Individual), f"Cannot compare with {type(object)} type"

    def __gt__(self, object) -> bool:
        self.__assert_type(object)
        return self.fitness > object.fitness

    def __str__(self) -> str:
        return f"Individual(path: {self.path[0]}...{self.path[-1]}, distance: {self.distance})"

    def __repr__(self) -> str:
        return str(self)
