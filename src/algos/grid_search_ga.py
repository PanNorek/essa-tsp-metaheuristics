from sklearn.model_selection import ParameterGrid
from . import GeneticAlgorithm
import pandas as pd
from ..utils import Result
from joblib import Parallel, delayed


class GridSearchGA:
    """
    GridSearchGA is a class that implements a grid search for the genetic algorithm.

    Parameters
    ----------

    param_grid: dict
        A dictionary with the parameters to be tested.
        The keys must be the same as the parameters of the GeneticAlgorithm class.
        The values must be a list of the values to be tested.

    n_jobs: int, default=-1
        The number of jobs to run in parallel.
        -1 means using all processors.

    Attributes
    ----------

    _param_grid: ParameterGrid
        A ParameterGrid object with the parameters to be tested.

    _n_jobs: int
        The number of jobs to run in parallel.

    _best_result: Result
        The best result found.

    _results: List[Result]
        A list with all the results found.
    """

    def __init__(self, param_grid: dict, n_jobs: int = -1) -> None:
        assert isinstance(param_grid, dict), "param_grid must be a dict"
        self._param_grid = ParameterGrid(param_grid)
        self._n_jobs = n_jobs
        self._best_result = None
        self._results = None

    def solve(self, distances: pd.DataFrame, return_best: bool = True) -> Result:
        print(f"GridSearchGA: {len(self._param_grid)} combinations were composed")
        # TODO: add progress bar maybe?
        self._results = Parallel(n_jobs=self._n_jobs)(
            delayed(GridSearchGA.solver)(distances, **params)
            for params in self._param_grid
        )
        self._best_result = min(self._results)

        return self._best_result if return_best else self._results

    @staticmethod
    def solver(distances, **kwargs):
        return GeneticAlgorithm(
            pop_size=kwargs["POP_SIZE"],
            n_iters=kwargs["N_ITERS"],
            selection_method=kwargs["SELECTION_METHOD"],
            crossover_method=kwargs["CROSSOVER_METHOD"],
            elite_size=kwargs["ELITE_SIZE"],
            mating_pool_size=kwargs["MATING_POOL_SIZE"],
            mutation_rate=kwargs["MUTATION_RATE"],
            neigh_type=kwargs["NEIGH_TYPE"],
            verbose=kwargs["VERBOSE"],
        ).solve(distances)
