import pandas as pd
from ..utils import StopAlgorithm
from .switching_algorithm import SwitchingAlgorithm
from .iterating_algorithm import IteratingAlgorithm


class HillClimbing(SwitchingAlgorithm, IteratingAlgorithm):
    """
    Hill Climber Algorithm

    Methods:
        solve - used for solving TSP problem

    Properties:
        best_path - best path found by algorithm
        history - list of best paths from each iteration

    Implements:
        SwitchingAlgorithm - wrapper around NeighbourhoodType,
            facilitates searching through adjacent solutions and gives
            more explicit comments in verbose mode
        IteratingAlgorithm - provides method to facilitates and order
            iterative approach to solving TSP with use of object attributes

    The hill climbing algorithm is one of the most naive approaches in solving the
    TSP. A random solution, a random sequence of cities, is generated. Then, all
    successor states of the solution is evaluated, where a successor state is obtained
    by switching the ordering of two cities adjacent in the solution,
    which is described in TSPAlgorithm as "searching neighbouring solutions space"
    The best successor is chosen only if it gives more optimal solution (shortest path),
    in other case the algorithm is finished.

    Setting start_order (options provided by TSPAlgorithm interface)
    in HillClimbing doesn't make sense, beacuse in case of lack
    of more optimal solutions in the vicinity, algorithm finishes at once
    giving the same result.

    One of the problems of this algorithm is possibility of getting stuck in
    local minimum, that's why is usually run multiple times in different vicinities
    (different random paths are chosen as starting point)

    Checks out src.algos.multistart_algorithm MultistartAlgorithm for the implementation.
    It enables running all algorithms with multistart in parallel using multiple threads

    For more information checks out:
    https://classes.engr.oregonstate.edu/mime/fall2017/rob537/hw_samples/hw2_sample2.pdf
    """

    NAME = "HILL CLIMBER"

    def _run_iteration(self, distances: pd.DataFrame) -> None:
        # number of iteration is increased by one
        self._next_iter()
        # get new path - best solution in vicinity
        new_path = self._switch(distances=distances, how='best')
        # get new distance - distance of the best solution in vicinity
        new_distance = self._get_path_distance(
            path=new_path, distances=distances
        )
        # distance gain
        gain = self.history[-1] - new_distance
        # break condition - if there are no more optimal solutions in vicinity
        # algorithm finishes
        if gain <= 0:
            raise StopAlgorithm(iteration=self._i, distance=new_distance)
        # in verbose mode information about switch that led to new solution is printed out
        if self._verbose:
            print(f"best switch: {self._last_switch_comment} - gain: {gain}")
            print(f"step {self._i}: distance: {new_distance}")

        # new path that shortens the distance
        self._path = new_path
        # adding new best distance to distances history
        self.history.append(new_distance)
