import random
import pandas as pd
from typing import Union, List, Tuple
from ..distance import get_path_distance
from .individual import Individual
from ..neighbourhood_type import NeighbourhoodType
from .parent_selection import ParentSelection, TruncationSelection, Tournament, Roulette


class Population:
    def __init__(self, pop_size: int = 500) -> None:
        self._pop_size = pop_size

    def generate_population(self, distances: pd.DataFrame) -> List[Individual]:
        """Generate a population of individuals

        Args:
            distances (pd.DataFrame): matrix of distances between cities

        Returns:
            List[Individual]: list of sampled individuals
        """
        indices = distances.index.to_list()
        paths = [random.sample(indices, len(indices)) for _ in range(self._pop_size)]
        distances = [get_path_distance(path=path, distances=distances) for path in paths]
        self._population = [
            Individual(path=path, distance=distance) for path, distance in zip(paths, distances)
        ]
        self.sort()

    @property
    def population(self) -> List[Individual]:
        return self._population

    @property
    def best(self) -> Individual:
        return max(self._population)

    def sort(self):
        self._population.sort(reverse=True)

    def crossover(self,
                  distances: pd.DataFrame,
                  sample_size: int | float,
                  crossover_method: ParentSelection,
                  crossover_rate: float,
                  elite_size: int = 0
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
            # TODO: what is that?!
            if random.random() < crossover_rate:
                # select two parents
                # TODO: idk
                parent1, parent2 = crossover_method.select(
                    self._population[elite_size:], size=sample_size)
                # crossing-over
                child = self._crossover(distances, parent1, parent2)
                # add child to population
                new_population.append(child)

        self._population = new_population
        self.sort()

    def mutate(self,
               distances: pd.DataFrame,
               neigh_type: NeighbourhoodType,
               skip: int | float = 0,
               mutation_rate: float = 0.5
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
            individual.distance = get_path_distance(path=individual.path,
                                                    distances=distances)
        self.sort()

    def _crossover(self,
                   distances: pd.DataFrame,
                   parent1: Individual,
                   parent2: Individual
                   ) -> Individual:
        """PMX (Partially Matched Crossover) algorithm for TSP problem.

        Args:
            parent1 (Individual): First parent
            parent2 (Individual): Second parent

        Returns:
            Individual: Child
        """
        # select a random subset of parent1
        start = random.randint(0, len(parent1.path) - 1)
        end = random.randint(start, len(parent1.path) - 1)
        subset = parent1.path[start:end]
        # create a child from the subset
        child = subset
        # add the remaining genes from parent2
        for gene in parent2.path:
            if gene not in subset:
                child.append(gene)
        # at this point we have parent1 genes at the beginning and the rest from parent2

        # recombine the child - put parent1 genes in the middle
        recombined_child = child[start:end] + child[:start] + child[end:]

        return Individual(path=recombined_child,
                          distance=get_path_distance(path=child, distances=distances))

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
