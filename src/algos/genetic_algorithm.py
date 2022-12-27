from typing import Union
import pandas as pd
from .iterating_algorithm import IteratingAlgorithm
from ..utils.genetic import (
    Population,
    TruncationSelection,
    Roulette,
    Tournament,
    PMX,
    OX,
)
from ..utils import Result, solve_it


class GeneticAlgorithm(IteratingAlgorithm):
    """Genetic algorithm for solving TSP problem"""

    NAME = "GENETIC ALGORITHM"
    DEFAULT_ITERS = 10
    _SELECTION_METHODS = {
        "truncation": TruncationSelection,
        "roulette": Roulette,
        "tournament": Tournament,
    }
    _CROSSOVER_METHODS = {"pmx": PMX, "ox": OX}

    def __init__(
        self,
        pop_size: int = 100,
        n_iters: int = DEFAULT_ITERS,
        selection_method: str = "truncation",
        crossover_method: str = "pmx",
        elite_size: Union[int, float] = 0,
        mating_pool_size: Union[int, float] = 0.5,
        mutation_rate: float = 0.5,
        neigh_type: str = "swap",
        verbose: bool = False,
    ) -> None:
        super().__init__(neigh_type=neigh_type, verbose=verbose, n_iter=n_iters)
        self._pop_size = pop_size
        self.population_ = None
        self._selection_method = selection_method
        self._crossover_method = crossover_method
        self._mutation_rate = mutation_rate
        self._elite_size = elite_size
        self._mating_pool_size = mating_pool_size
        self.mean_distances = []
        self.__check_params()

    def __check_params(self):
        assert 0 <= self._mutation_rate <= 1, "Mutation rate must be between 0 and 1."
        assert (
            self._selection_method in self._SELECTION_METHODS
        ), f"selection method must be one of {self._SELECTION_METHODS}"
        assert (
            self._crossover_method in self._CROSSOVER_METHODS
        ), f"crossover method must be one of {self._CROSSOVER_METHODS}"
        assert isinstance(
            self._elite_size, (int, float)
        ), "elite size must be int or float"
        self._crossover = self._CROSSOVER_METHODS[self._crossover_method]()

    @solve_it
    def _solve(
        self,
        distances: pd.DataFrame,
        random_seed: Union[int, None] = None,
        start_order: Population = None,
    ) -> pd.DataFrame:
        # 1st stage: Create random population
        self.population_: Population = self._setup_start(
            distances=distances, random_seed=random_seed, start_order=start_order
        )
        # 2nd stage: Loop for each generation
        for _ in range(self._n_iter):
            self._run_iteration(distances=distances)

        result = Result(
            algorithm=self,
            path=self.path_,
            distance=self.history[-1],
            distance_history=self.history,
            mean_distance_history=self.mean_distances,
        )
        return result

    def _run_iteration(self, distances: pd.DataFrame) -> None:
        self._next_iter()
        # I: Crossover - make children
        self.population_.crossover(
            distances=distances,
            sample_size=self._mating_pool_size,
            selection_method=self._SELECTION_METHODS[self._selection_method],
            crossover_method=self._crossover,
            elite_size=self._elite_size,
        )
        # II: Mutation - mutate all population
        self.population_.mutate(
            distances=distances,
            neigh_type=self._neigh,
            skip=self._elite_size,
            mutation_rate=self._mutation_rate,
        )
        self.history.append(self.population_.best.distance)
        self._path = self.population_.best.path
        self.mean_distances.append(self.population_.mean_distance)

        if self._verbose:
            print(
                f"Generation {self._i} best distance: {self.population_.best.distance:.2f}"
            )
            print(
                f"Generation {self._i} mean distance: {self.population_.mean_distance:.2f}"
            )

    def _set_start_order(
        self, distances: pd.DataFrame, start_order: Population = None
    ) -> Population:
        if start_order is not None:
            self._start_order_check(start_order=start_order)
            return start_order
        population = Population(pop_size=self._pop_size)
        population.generate_population(distances=distances)
        return population

    def _start_order_check(self, start_order: Population) -> None:
        assert isinstance(start_order, Population), "start_order must be iterable"
        assert (
            len(start_order) == self._pop_size
        ), f"Expected population with {self._pop_size} elements, got {len(start_order)}"

    def __str__(self) -> str:
        mes = super().__str__()
        # replace __class__.__name__ with __str__ methods in crossover
        mes += f"""\npop_size: {self._pop_size}\n\
        generations: {self._n_iter}\n\
        selection_method: {self._selection_method}\n\
        crossover_method: {self._crossover.__class__.__name__}\n\
        elite_size: {self._elite_size}\n\
        mating_pool_size: {self._mating_pool_size}\n\
        mutation_rate: {self._mutation_rate}\n"""
        return mes
