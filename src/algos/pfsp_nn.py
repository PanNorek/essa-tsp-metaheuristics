from src.algos import TSPAlgorithm
from typing import Union, Any
import pandas as pd
from src.utils import solve_it, Result, get_order_cost
import random


class PFSP_NearestNeighbor(TSPAlgorithm):
    """Proposed nearest neighbor model for the permutation flow shop problem.
    We add successively more tasks to the initial task by locally minimizing the durations on the machines.

    Steps:
        1. Get the tasks
        2. Initialize the path
        3. Find the nearest neighbor
            3.1: Calculate the distances for all new switches
            3.2: Find the nearest neighbor
            3.3: Add the nearest neighbor to the path
            3.4: Remove the nearest neighbor from the unordered tasks
        4. Calculate the final cost function
        5. Return the result

    Example:
        1. Start with the task 1
        2. Calculate the cost for all possible switches (task 1 -> task 2, task 1 -> task 3, ...)
        3. Choose the task with the lowest cost -> lowest time to end all tasks on all machines
        4. Add the task to the path
        5. Repeat steps 2-4 until all tasks are added to the path

    """

    NAME = "PFSP_NearestNeighbor"

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(verbose)

    @solve_it
    def _solve(
        self,
        distances: pd.DataFrame,
        start_order: Union[int, None] = None,
    ) -> Result:

        # Step I: Get the tasks
        unordered_tasks = distances.index.to_list()

        # Step I.1: Check if start_order is specified
        if not start_order:
            start_order = random.choice(unordered_tasks)

        unordered_tasks.remove(start_order)
        # Step II: Initialize the path
        self._path = [start_order]
        # Step III: Find the nearest neighbor
        while unordered_tasks:
            # Step III.1: Calculate the distances for all new switches
            possibilites = {
                task_number: get_order_cost(self._path + [task_number], distances)
                for task_number in unordered_tasks
            }
            # Step III.2: Find the nearest neighbor
            nearest_neighbor = min(possibilites, key=possibilites.get)
            # Step III.3: Add the nearest neighbor to the path
            self._path.append(nearest_neighbor)
            # Step III.4: Remove the nearest neighbor from the unordered tasks
            unordered_tasks.remove(nearest_neighbor)
        # Step IV: Calculate the final cost function
        distance = self._get_path_distance(path=self._path, distances=distances)
        # Step V: Return the result
        return Result(algorithm=self, path=self._path, distance=distance)
