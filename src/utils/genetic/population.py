import random
import pandas as pd
from typing import Union, List, Tuple
from .individual import Individual
from .mutable import Mutable, SimpleSwap


class Population:
    DIST_FITNESS = "cum_fitness"
    # TODO: Implement init method

    def __init__(self, pop_size: int = 500) -> None:
        self._pop_size = pop_size

    def generate_population(self, distances: pd.DataFrame) -> List[Individual]:
        """Generate a population of individuals

        Args:
            distances (pd.DataFrame): _description_

        Returns:
            List[Individual]: _description_
        """
        indices = distances.index.to_list()
        paths = [random.sample(indices, len(indices)) for _ in range(self._pop_size)]
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

    def select(self, method: str = "elitism") -> List[Individual]:
        if method == "elitism":
            # select the best half of the population
            self._select_elitism()
        elif method == "roulette":
            self._select_roulette()

    def crossover(
        self,
        distances: pd.DataFrame,
        crossover_rate: float = 0.9,
    ) -> None:
        # at this point, the population is 1/2 of the original size (selection worked)
        assert len(self.population) == self._pop_size // 2
        # order the population by fitness
        self._order_population()  # TODO: is this necessary?
        # create a copy of the population
        generation = self.population.copy()
        # while the population is not full
        while len(self.population) < self._pop_size:
            # check if crossover should occur
            if random.random() < crossover_rate:
                # select two parents
                parent1, parent2 = random.sample(generation, 2)
                # crossing-over
                child = self._crossover(distances, parent1.path, parent2.path)
                # add child to population
                self._population.append(child)

    def mutate(
        self,
        distances: pd.DataFrame,
        mutation_type: Mutable,
        mutation_rate: float = 0.5,
    ) -> None:

        for individual in self.population:
            individual.mutate(mutation_type, mutation_rate)
            individual.distance = self._get_path_distance(distances, individual.path)
        self._order_population()

    def _order_population(self):
        self._population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

    def _crossover(
        self, distances: pd.DataFrame, parent1: Individual, parent2: Individual
    ) -> Individual:
        """Order 1 crossover

        Args:
            parent1 (Individual): _description_
            parent2 (Individual): _description_

        Returns:
            Individual: _description_
        """
        # select a random subset of parent1
        start = random.randint(0, len(parent1) - 1)
        end = random.randint(start, len(parent1) - 1)
        subset = parent1[start:end]
        # create a child from the subset
        child = subset
        for gene in parent2:
            if gene not in subset:
                child.append(gene)
        return Individual(path=child, distance=self._get_path_distance(distances, child))

    def _select_roulette(self, generation: List[Individual]) -> Individual:
        """Select an individual using roulette wheel selection"""
        # define random threshold
        pick = random.random()
        mask = generation[self.DIST_FITNESS] >= pick
        return generation.loc[mask, Individual.INDIVIDUAL].iloc[0]

    def _select_elitism(self):
        self._population = self._population[: (len(self.population) // 2)]

    def _add_weights(self, generation: List[Individual]) -> pd.DataFrame:
        """Add weights to the population (shorter paths have higher weights)"""
        # convert to dataframe
        df = pd.concat([ind.to_df() for ind in generation], ignore_index=True)
        # add weights
        cum_fit = df[Individual.FITNESS].cumsum()
        df[self.DIST_FITNESS] = cum_fit.div(df[Individual.FITNESS].sum())  # FIXME: CHECK THIS
        return df

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
