from typing import Union, List, Callable
import pandas as pd
import numpy as np
from . import Algorithm
from ..utils.genetic import (
    Population,
    TruncationSelection,
    Roulette,
    Tournament,
    PMX,
    OX
)
from ..utils import Result, time_it


class GeneticAlgorithm(Algorithm):
    """Genetic algorithm for solving TSP problem"""

    NAME = "GENETIC ALGORITHM"
    _SELECTION_METHODS = {
        "truncation": TruncationSelection,
        "roulette": Roulette,
        "tournament": Tournament
    }
    _CROSSOVER_METHODS = {
        "pmx": PMX,
        "ox": OX
    }

    def __init__(
        self,
        pop_size: int = 100,
        no_generations: int = 10,
        selection_method: str = "truncation",
        crossover_method: str = "pmx",
        elite_size: Union[int, float] = 0,
        mating_pool_size: Union[int, None] = None,
        mutation_rate: float = 0.5,
        neigh_type: str = "swap",
        verbose: bool = False,
        inversion_window: int | None = None
    ):
        super().__init__(neigh_type=neigh_type,
                         verbose=verbose,
                         inversion_window=inversion_window)
        self._pop_size = pop_size
        self.no_generations = no_generations
        self._selection_method = selection_method
        self._crossover_method = crossover_method
        self._mutation_rate = mutation_rate
        self._history = [np.inf]
        self._mean_distance = []
        self._elite_size = elite_size
        self.__check_params()

    def __check_params(self):
        assert 0 <= self._mutation_rate <= 1, "Mutation rate must be between 0 and 1."
        assert (
            self._selection_method in self._SELECTION_METHODS
        ), f"selection method must be one of {self._SELECTION_METHODS}"
        assert (
            self._crossover_method in self._CROSSOVER_METHODS
        ), f"crossover method must be one of {self._CROSSOVER_METHODS}"
        self._crossover = self._CROSSOVER_METHODS[self._crossover_method]()

    @time_it
    def solve(self,
              distances: pd.DataFrame,
              random_seed: Union[int, None] = None
              ) -> pd.DataFrame:
        super().solve(distances=distances, random_seed=random_seed)
        # 1st stage: Create random population
        population = Population(pop_size=self._pop_size)
        population.generate_population(distances=distances)

        # 2nd stage: Loop for each generation
        for _ in range(self.no_generations):
            print(_)
            # I: Crossover - make children
            population.crossover(
                distances=distances,
                # TODO adjust - mating_pool_size
                sample_size=0.5,
                selection_method=self._SELECTION_METHODS[self._selection_method],
                crossover_method=self._crossover,
                elite_size=self._elite_size
            )
            # II: Mutation - mutate all population
            population.mutate(
                distances=distances,
                neigh_type=self._neigh,
                skip=self._elite_size,
                mutation_rate=self._mutation_rate,
            )
            # story only better results
            if population.best.distance < self._history[-1]:
                self._history.append(population.population[0].distance)

        return Result(
            algorithm=self,
            path=population.best.path,
            best_distance=population.best.distance,
            distance_history=self.history
        )
