import time
import pandas as pd
from joblib import Parallel, delayed
from .switching_algorithm import SwitchingAlgorithm
from ..utils import Result


def solver(algorithm: SwitchingAlgorithm, distances: pd.DataFrame):
    """Solver function for parallel processing"""
    return algorithm.solve(distances=distances)


class MultistartAlgorithm:
    """Multistart Algorithm

    Call this class with algorithm and distance matrix
    to run algorithm with multiple starts in parallel.
    """

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def __call__(
        self,
        algorithm: SwitchingAlgorithm,
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
        algo: SwitchingAlgorithm = algorithm(**kwargs)

        tic = time.time()
        results: list[Result] = Parallel(n_jobs=n_jobs)(
            delayed(solver)(algorithm=algo, distances=distances)
            for _ in range(n_starts)
        )
        toc = time.time()

        if self._verbose:
            print(
                f"Solving time for {algorithm.NAME}: {toc - tic:.3f} s - Parallel: {n_jobs != 1}"
            )

        if only_best:
            return min(results)

        results_df = pd.DataFrame().from_dict([x.to_dict() for x in results])
        results_df = results_df.sort_values()
        return results_df
