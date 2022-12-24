from sklearn.model_selection import ParameterGrid
from . import GeneticAlgorithm
import pandas as pd
from ..utils import Result, solve_it
from joblib import Parallel, delayed


class GridSearchGA:
    def __init__(self, param_grid: dict, n_jobs: int = -1) -> None:
        assert isinstance(param_grid, dict), "param_grid must be a dict"
        self._param_grid = ParameterGrid(param_grid)
        self._n_jobs = n_jobs

    @solve_it
    def solve(self, distances: pd.DataFrame) -> Result:
        print(f"GridSearchgGA: {len(self._param_grid)} cobinations were composed")
        results = Parallel(n_jobs=self._n_jobs)(
            delayed(GridSearchGA.solver)(distances, **params)
            for params in self._param_grid
        )
        return min(results, key=lambda x: x.best_distance)

    @staticmethod
    def solver(distances, **kwargs):
        return GeneticAlgorithm(
            pop_size=kwargs["POP_SIZE"],
            no_generations=kwargs["NO_GENERATIONS"],
            selection_method=kwargs["SELECTION_METHOD"],
            crossover_method=kwargs["CROSSOVER_METHOD"],
            elite_size=kwargs["ELITE_SIZE"],
            mating_pool_size=kwargs["MATING_POOL_SIZE"],
            mutation_rate=kwargs["MUTATION_RATE"],
            neigh_type=kwargs["NEIGH_TYPE"],
            verbose=kwargs["VERBOSE"],
        ).solve(distances)
