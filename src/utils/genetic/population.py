import random
import pandas as pd
from typing import Union, List
from .individual import Individual


class Population:
    DIST_FITNESS = 'cum_fitness'

    def __init__(self, distances: pd.DataFrame, pop_size: int) -> None:
        self._distances = distances
        self._pop_size = pop_size
        self._population = self._generate_population()

    def _generate_population(self) -> List[Individual]:
        indices = self._distances.index.to_list()
        paths = [random.sample(indices, len(indices))
                 for _ in range(self._pop_size)]
        distances = [self._get_path_distance(path=path) for path in paths]
        population = [Individual(path=path, distance=distance)
                      for path, distance in zip(paths, distances)]
        return population

    def _get_path_distance(self, path: list) -> int:
        path_length = sum([self._distances.loc[x, y]
                           for x, y in zip(path, path[1:])])
        # add distance back to the starting point
        path_length += self._distances.loc[path[0], path[-1]]
        return path_length

    def select(self, method: str = 'roulette'):
        return [self._select_roulette() for _ in range(2)]

    def _select_roulette(self):
        pick = random.random()
        df = self._to_df()
        mask = df[self.DIST_FITNESS] >= pick
        parent = df.loc[mask, Individual.INDIVIDUAL].iloc[0]
        return parent

    def _to_df(self) -> pd.DataFrame:
        df = pd.concat([ind.to_df() for ind in self], ignore_index=True)
        cum_fit = df[Individual.FITNESS].cumsum()
        df[self.DIST_FITNESS] = cum_fit.div(df[Individual.FITNESS].sum())
        return df

    @property
    def population(self) -> List[Individual]:
        return self._population

    def __str__(self) -> str:
        return str(self.population)

    def __repr__(self) -> str:
        return str(self)

    def __len__(self):
        return len(self.population)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self[i] for i in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f'The index {key} is out of range')
            return self.population[key]
        else:
            raise TypeError('Invalid argument type')

    def __iter__(self):
        self.__i = 0
        return self

    def __next__(self):
        if self.__i < len(self.population):
            element = self.population[self.__i]
            self.__i += 1
            return element
        else:
            raise StopIteration
