import random
import pandas as pd
from typing import Union, List
from ..tools import get_path_distance
from .individual import Individual
from ..neighbourhood_type import NeighbourhoodType
from .parent_selection import ParentSelection
from .crossover import CrossoverMethod


class Population:
    def __init__(self, pop_size: int = 500) -> None:
        self._pop_size = pop_size
        self._population = []

    def generate_population(self, distances: pd.DataFrame):
        """Generate a population of individuals

        Args:
            distances (pd.DataFrame): matrix of distances between cities

        Returns:
            List[Individual]: list of sampled individuals
        """
        indices = distances.index.to_list()
        paths = [random.sample(indices, len(indices)) for _ in range(self._pop_size)]
        distances = [
            get_path_distance(path=path, distances=distances) for path in paths
        ]
        self._population = [
            Individual(path=path, distance=distance)
            for path, distance in zip(paths, distances)
        ]
        self.sort()
        return self

    @property
    def mean_distance(self) -> float:
        return sum(ind.distance for ind in self._population) / len(self._population)

    @property
    def population(self) -> List[Individual]:
        return self._population

    @property
    def best(self) -> Individual:
        return max(self._population)

    def sort(self):
        self._population.sort(reverse=True)

    def crossover(
        self,
        distances: pd.DataFrame,
        sample_size: Union[int, float],
        selection_method: ParentSelection,
        crossover_method: CrossoverMethod,
        elite_size: int = 0,
    ) -> None:
        """Crossover the population.
        Function uses PMX (Partially Matched Crossover) algorithm.

        Args:
            distances (pd.DataFrame): matrix of distances between cities
            crossover_rate (float): probability of crossover
        """
        new_population = self._population[:elite_size]
        # while the population is not full
        while len(new_population) < self._pop_size:
            # check if crossover should occur
            parent_1, parent_2 = selection_method.select(
                self._population[elite_size:], size=sample_size
            )

            # crossing-over
            child_1, child_2 = crossover_method.crossover(
                parent_1=parent_1.path, parent_2=parent_2.path
            )
            child_1, child_2 = (
                Individual(
                    path=child_1,
                    distance=get_path_distance(path=child_1, distances=distances),
                ),
                Individual(
                    path=child_2,
                    distance=get_path_distance(path=child_2, distances=distances),
                ),
            )
            # add child to population
            new_population.append(child_1)
            if len(new_population) < self._pop_size:
                new_population.append(child_2)

        self._population = new_population
        self.sort()

    def mutate(
        self,
        distances: pd.DataFrame,
        neigh_type: NeighbourhoodType,
        skip: Union[int, float] = 0,
        mutation_rate: float = 0.5,
    ) -> None:
        """Mutate the population

        Args:
            distances (pd.DataFrame): matrix of distances between cities
            mutation_type (Mutable): mutation type. Possible values:
            mutation_rate (float, optional): probability of mutation. Defaults to 0.5.
        """
        if isinstance(skip, float):
            skip = int(len(self._population) * skip)

        for individual in self._population[skip:]:
            if random.random() > mutation_rate:
                continue
            individual.mutate(neigh_type=neigh_type)
            individual.distance = get_path_distance(
                path=individual.path, distances=distances
            )
        self.sort()

    # magic methods
    # --------------

    def __str__(self) -> str:
        return f"Size: {len(self)}\n best: {self.best}"

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
                raise IndexError(f"The index {key} is out of range")
            return self.population[key]
        else:
            raise TypeError("Invalid argument type")

    def __iter__(self):
        self.__i = 0
        return self

    def __next__(self):
        if self.__i >= len(self.population):
            raise StopIteration
        element = self.population[self.__i]
        self.__i += 1
        return element
