import pandas as pd
from sklearn.model_selection import ParameterGrid
from .multistart_algorithm import MultistartAlgorithm
from .algorithm import TSPAlgorithm
from ..utils import Result
from typing import Union


class TSPGridSearch:
    """Exhaustive search over specified parameter values for an algorithm

    Methods:
        solve - Run solve method with all sets of parameters

    Uses MultistartAlgorithm to run the same algorithm multiple times.

    Example:

    gs = TSPGridSearch()
    gs.solve(algorithm=HillClimbing,
             params={"neigh_type": ["swap", "inversion"], "n_iters": [20,30]},
             distances=distances)

    Check out:

    src.algos.multistart_algorithm MultistartAlgorithm
    """

    def solve(self,
              algorithm: type[TSPAlgorithm],
              params: dict,
              distances: pd.DataFrame,
              n_starts: int = 5,
              only_best: bool = True,
              verbose: bool = True,
              n_jobs: int = 1
              ) -> Union[Result, pd.DataFrame]:

        """
        Run solve method with all sets of parameters

        Params:
            algorithm: type[TSPAlgorithm]
                Algorithm to run
            params: dict
                Dictionary of all algorithm params with its values
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns
            n_starts int:
                Number of starts
            only_best: bool
                If False returns DataFrame of n_starts results,
                by default True, returns only the best result
            verbose: bool
                If True, prints out information about grid search progress
            n_jobs: int
                Number of thread to involve

        algorithm:
            TSPAlgorithm subclass itself rather than its instance

        n_jobs:
            Deafult 1, a standard synchronous run
            Set to -1 to run on all available threads
        """
        # iterable object
        search_grid = ParameterGrid(params)

        if verbose:
            print(f"number of algorithms in grid search: {len(search_grid)}")

        multistart = MultistartAlgorithm(verbose=verbose)
        # if only_best=True it's a list of Results
        # otherwise - a list of pd.DataFrame
        results: Union[list[Result], list[pd.DataFrame]] = [
            multistart(
                algorithm=algorithm,
                distances=distances,
                n_starts=n_starts,
                only_best=only_best,
                n_jobs=n_jobs,
                **param_set
            )
            for param_set in search_grid]

        if only_best:
            return min(results)

        df_results = pd.concat(results, ignore_index=True)
        return df_results
