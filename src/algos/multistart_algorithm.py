import time
import pandas as pd
from joblib import Parallel, delayed
from .swapping_algorithm import SwappingAlgorithm
from ..utils import Result


def solver(algo, distances, **kwargs):
    """Solver function for parallel processing"""
    return algo(**kwargs).solve(distances)


class MultistartAlgorithm:
    """Multistart Algorithm

    Call this class with algorithm and distance matrix
    to run algorithm with multiple starts in parallel.
    """

    def __call__(
        self,
        algorithm: SwappingAlgorithm,
        distances: pd.DataFrame,
        n_starts=10,
        only_best=True,
        n_jobs: int = -1,
        **kwargs,
    ) -> Result:
        """Run algorithm with multiple starts in parallel
        Args:

            algorithm (SwappingAlgorithm): Algorithm to run
            distances (pd.DataFrame): Distance matrix
            n_starts (int, optional): Number of starts. Defaults to 10.
            only_best (bool, optional): Return only best result. Defaults to True.
            n_jobs (int, optional): Number of jobs. Defaults to -1. -1 means all available.
            **kwargs: Keyword arguments for algorithm

        Returns:
            Result: Result of algorithm
        """
        tic = time.time()
        algorithms = [algorithm for _ in range(n_starts)]
        results_parallel = Parallel(n_jobs=n_jobs)(
            delayed(solver)(algo, distances, **kwargs) for algo in algorithms
        )
        toc = time.time()
        print(f"Parallel time for {algorithm.NAME}: {toc - tic:.3f} s")

        if only_best:
            return min(results_parallel)
        results_df = pd.DataFrame().from_dict([x.__dict__ for x in results_parallel])
        results_df["algorithm"] = results_df["algorithm"].apply(lambda x: x.NAME)
        return results_df
