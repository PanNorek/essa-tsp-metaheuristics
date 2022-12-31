import pandas as pd
from .iterating_algorithm import IteratingAlgorithm
from ..utils import Queue


class TabuSearch(IteratingAlgorithm):
    """
    Tabu Search Algorithm

    Methods:
        solve - used for solving TSP problem

    Properties:
        path_ - best path found by algorithm
        history - list of best paths from each iteration

    Implements:
        IteratingAlgorithm - provides method to facilitates and order
            iterative approach to solving TSP with use of object attributes

    Tabu search is a metaheuristic search method employing
    local search methods used for mathematical optimization
    Local (neighborhood) searches take a potential solution to a problem
    and check its immediate neighbors
    (that is, solutions that are similar except for very few minor details)
    in the hope of finding an improved solution.
    Local search methods have a tendency to become stuck in suboptimal
    regions or on plateaus where many solutions are equally fit.
    (check out HillClimbing documentation)
    Tabu search enhances the performance of local search by relaxing its basic rule.
    First, at each step worsening moves can be accepted if no improving move is available
    (like when the search is stuck at a strict local minimum).
    In addition, prohibitions (henceforth the term tabu) are introduced to discourage
    the search from coming back to previously-visited solutions.
    If a potential solution has been previously visited within a certain short-term period
    it is marked as "tabu" (forbidden) so that the algorithm
    does not consider that possibility repeatedly.

    Note:
        Longer runs usually doesn't improve the solution significantly
        but are taking more time

    For more information check out:
    https://en.wikipedia.org/wiki/Tabu_search
    """

    DEFAULT_ITERS = 100
    NAME = "TABU SEARCH"

    def __init__(
        self,
        tabu_length: int = 3,
        neigh_type: str = "swap",
        n_iter: int = DEFAULT_ITERS,
        verbose: bool = False,
    ) -> None:
        """
        Params:
            tabu_length: int
                Number of recent elements in vicinity forbidden to choose from
            neigh_type: str
                Type of neighbourhood used in algorithm
            n_iter: int
                Max number of iterations to run before finishing the algorithm
            verbose: bool
                If True, prints out information about algorithm progress

        neigh_type:
            "swap": swapping two elements in a list
            "inversion": inversing order of a slice of a list
            "insertion": inserting element into a place
        """
        super().__init__(neigh_type=neigh_type, n_iter=n_iter, verbose=verbose)
        self._tabu_list = Queue(length=tabu_length)

    def _run_iteration(self, distances: pd.DataFrame) -> None:
        # number of iteration is increased by one
        self._next_iter()
        # get new path - best solution in vicinity
        new_path = self._switch(
            distances=distances, how="best", exclude=self._tabu_list
        )
        # get new distance - distance of the best solution in vicinity
        new_distance = self._get_path_distance(path=new_path, distances=distances)
        # new path that minimizes distance - always accepts
        # new optimal solution in vicinity even if it's worse
        # a way not to stuck in local minimum
        self._path = new_path
        # removing first switch from tabu list only if list is full
        self._tabu_list.dequeue()
        # adding new switch to tabu list
        self._tabu_list.enqueue(self._last_switch)
        # in verbose mode information about switch that led to new solution is printed out
        if self._verbose:
            # distance gain
            gain = self._history[-1] - new_distance
            print(f"best switch: {self._last_switch_comment} - gain: {gain}")
            print(f"step {self._i}: distance: {new_distance}")
            print(f"tabu list: {self._tabu_list}")

        # adding new best distance to distances history
        self._history.append(new_distance)
