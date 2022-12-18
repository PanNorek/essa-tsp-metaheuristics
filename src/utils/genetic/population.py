import random
import pandas as pd
from typing import List
from .individual import Individual
from .mutable import Mutable
from .parent_selection import ParentSelection


class Population:
    def __init__(self, init_pop_size: int, pop_size: int) -> None:
        self._init_pop_size = init_pop_size
        self._pop_size = pop_size

    def generate_population(self, distances: pd.DataFrame) -> List[Individual]:
        """Generate a population of individuals

        Args:
            distances (pd.DataFrame): matrix of distances between cities

        Returns:
            List[Individual]: list of sampled individuals
        """
        indices = distances.index.to_list()
        paths = [random.sample(indices, len(indices)) for _ in range(self._init_pop_size)]
        distances = [self._get_path_distance(distances=distances, path=path) for path in paths]
        self._population = [
            Individual(path=path, distance=distance) for path, distance in zip(paths, distances)
        ]
        self._population = sorted(self._population, key=lambda x: x.fitness, reverse=True)

    def _get_path_distance(self, distances: pd.DataFrame, path: list) -> int:
        """Get the distance of a given path"""
        path_length = sum(distances.loc[x, y] for x, y in zip(path, path[1:]))
        # add distance back to the starting point
        path_length += distances.loc[path[0], path[-1]]
        return path_length

    @property
    def population(self) -> List[Individual]:
        return self._population

    def crossover(
        self,
        distances: pd.DataFrame,
        crossover_method: ParentSelection,
        crossover_rate: float,
        **kwargs,
    ) -> None:
        """Crossover the population.
        Function uses PMX (Partially Matched Crossover) algorithm.

        Args:
            distances (pd.DataFrame): matrix of distances between cities
            crossover_rate (float): probability of crossover
        """

        assert len(self.population) < self._pop_size, "Population is full."

        # sort the population by fitness
        self._order_population()

        # create a copy of the population
        generation = self.population.copy()

        # while the population is not full
        while len(self.population) < self._pop_size:
            # check if crossover should occur
            if random.random() < crossover_rate:
                # select two parents
                parent1, parent2 = crossover_method.select(generation, **kwargs)
                # crossing-over
                child = self._crossover(distances, parent1, parent2)
                # add child to population
                self._population.append(child)

    def mutate(
        self,
        distances: pd.DataFrame,
        mutation_type: Mutable,
        mutation_rate: float = 0.5,
    ) -> None:
        """Mutate the population

        Args:
            distances (pd.DataFrame): matrix of distances between cities
            mutation_type (Mutable): mutation type. Possible values: SimpleSwap, Inversion, Insertion.
            mutation_rate (float, optional): probability of mutation. Defaults to 0.5.
        """

        for individual in self.population:
            individual.mutate(mutation_type, mutation_rate)
            individual.distance = self._get_path_distance(distances, individual.path)
        self._order_population()

    def select(self) -> None:
        """Select the better half of the population"""
        # sort the population by fitness
        self._order_population()
        self._population = self._population[: self._init_pop_size]

    def calculate_mean_distance(self) -> float:
        """Calculate the mean distance of the population"""
        return sum(individual.distance for individual in self.population) / len(self.population)

    def _order_population(self):
        self._population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

    def _crossover(
        self, distances: pd.DataFrame, parent1: Individual, parent2: Individual
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

        return Individual(path=recombined_child, distance=self._get_path_distance(distances, child))

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
