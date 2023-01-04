import time
import pandas as pd
from joblib import Parallel, delayed
from typing import Union
from .algorithm import TSPAlgorithm
from ..utils import Result


def solver(algorithm: TSPAlgorithm, distances: pd.DataFrame):
    """Solver function for parallel processing"""
    return algorithm.solve(distances=distances)


class MultistartAlgorithm:
    """
    Multistart Algorithm

    Call this class with algorithm and distance matrix
    to run algorithm with multiple starts in parallel.
    """

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def __call__(
        self,
        algorithm: type[TSPAlgorithm],
        distances: pd.DataFrame,
        n_starts: int = 10,
        only_best: bool = True,
        n_jobs: int = -1,
        **kwargs,
    ) -> Union[Result, pd.DataFrame]:
        """Runs algorithm with multiple starts in parallel
            Params:

            algorithm: type[TSPAlgorithm]
                Algorithm to run
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns
            n_starts int:
                Number of starts
            only_best: bool
                If False returns DataFrame of n_starts results
            n_jobs: int
                Number of thread to involve
            **kwargs:
                Keyword arguments for algorithm init

        algorithm:
            TSPAlgorithm subclass itself rather than its instance

        n_jobs:
            Deafult -1, run on all available threads
            One means a standard synchronous run
        """
        # algorithm object - Parallel makes a copy of an object
        algo: TSPAlgorithm = algorithm(**kwargs)
        # prints out info about running algorithm in verbose mode
        if self._verbose:
            print(f"\nparams: {kwargs}")
            print(algo)
        # keeping track of time
        tic = time.time()
        results: list[Result] = Parallel(n_jobs=n_jobs)(
            delayed(solver)(
                algorithm=algo, distances=distances
            )
            for _ in range(n_starts)
        )
        toc = time.time()

        # prints out time in verbose mode
        if self._verbose:
            print(
                f"Solving time for {algorithm.NAME}: {toc - tic:.3f} s - Parallel: {n_jobs != 1}"
            )
        # returns the best result
        # note: all results are saved in csv
        if only_best:
            return min(results)

        # if only_best is False return df with all results
        results.sort()
        results_df = pd.DataFrame().from_dict([x.to_dict() for x in results])
        return results_df
