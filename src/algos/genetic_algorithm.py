from typing import Union, List, Callable
import pandas as pd
from . import Algorithm
from ..utils.genetic import Individual, Population
import random
import numpy as np


class GeneticAlgorithm(Algorithm):
    """ Genetic algorithm for solving TSP problem """

    def __init__(self,
                 pop_size: int,
                 no_generations: int,
                 selection_method: str = "elitism",
                 crossover_method: str = "ordered",
                 crossover_rate: float = 1.0,
                 mutation_rate: float = .5,
                 neigh_type: str = None,
                 random_state: int = None,
                 verbose: bool = False):
        super().__init__(neigh_type=neigh_type, verbose=verbose)
        self._pop_size = pop_size
        self.no_generations = no_generations
        self._selection_method = selection_method
        self._crossover_rate = crossover_rate
        self._crossover_method = crossover_method
        self._mutation_rate = mutation_rate
        # TODO: is this the same as np.random.seed(random_state)?
        random.seed(random_state)
        np.random.seed(random_state)

    def solve(self, distances: pd.DataFrame) -> pd.DataFrame:
        # 1st stage: Create random population population
        population = Population(self._pop_size)
        population.generate_population(distances)
        print(f"Initial population: {population.population[0]}")
        # 2nd stage: Loop for each generation
        for generation in range(self.no_generations):
            pass
            # I: Select parents
            population.select(method=self._selection_method)
            # II: Crossover - choose whether to give birth to a child or two
            population.crossover(
                distances, crossover_method=self._crossover_method, crossover_rate=self._crossover_rate)
            # III: Mutation
            population.mutate(distances, self._mutation_rate)
            # print(f"Generation {generation}: {population.population[0]}")
        print(f"Final population: {population.population[0]}")
