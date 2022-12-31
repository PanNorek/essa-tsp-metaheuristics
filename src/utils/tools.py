import pandas as pd
import time
from typing import Callable, Union, Any, Iterable
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import os
import random


@dataclass
class Result:
    """
    Class for storing the results of TSPAlgorithm

    Methods:
        to_dict:
            Transforms result into a dictionary
        to_df:
            Transforms result into a one-row pd.DataFrame

    Attributes:
        algorithm:
            TSPAlgorithm object that solved TSP problem
            giving this solution
        path:
            optimal solution found be the algorithm
            In TSP represents an order of cities that minimizes
            the distance
        distance:
            Distance to be traversed by the salesman with optimal
            path found be the algorithm. In TSP distance is a cost function
        time:
            Time of solving the problem
        distance_history:
            List of accepted solutions during iteration process
        mean_distance_history:
            Specific for GeneticAlgorithm, list of mean distance
            of each generation, used to show the progress of evolution

    Check out:

    src.utils.heuristic_algorithm TSPAlgorithm
    """

    algorithm: Any
    path: list
    distance: int
    time: float = None
    distance_history: list = None
    mean_distance_history: list = None

    def __str__(self) -> str:
        """How Result object is represented as string"""
        mes = f"""distance: {self.distance}
                  algorithm: {self.algorithm}
                  solving time: {self.time:.3f} s
                """.replace(
            "  ", ""
        )
        if self.distance_history:
            mes += f"""history: {self._str_history()}\n"""
        return mes

    def _str_history(self) -> str:
        """String representation of distance history"""
        # if distance history if too long, show only start and end of it
        if len(self.distance_history) > 10:
            return f"{self.distance_history[1]}, {self.distance_history[2]} ... " +\
                f"{self.distance_history[-2]}, {self.distance_history[-1]}"
        return self.distance_history

    def __repr__(self) -> str:
        """How Result object is represented as string"""
        return str(self)

    def __gt__(self, object_) -> bool:
        # specific implementation for TSP
        # shorter distance == better solution
        assert isinstance(
            object_, (Result, float, int)
        ), f"can't compare result with {type(object_)} type"
        # if compared to another Result object
        if isinstance(object_, Result):
            return self.distance > object_.distance
        # if compared to a numeric value
        return self.distance > object_

    def to_df(self) -> pd.DataFrame:
        """Transforms result into a one-row pd.DataFrame for better visualization"""
        df = pd.DataFrame(
            data={
                "algorithm": [self.algorithm],
                "path": [self.path],
                "distance": [self.distance],
                "solving_time": [self.time],
                "size": len(self.path),
            }
        )
        return df

    def to_dict(self) -> dict:
        """Transforms result into a dictionary"""
        return {
            "algorithm": self.algorithm,
            "path": self.path,
            "distance": self.distance,
            "solving_time": self.time,
            "size": len(self.path),
        }


def get_path_distance(path: list, distances: pd.DataFrame) -> Union[int, float]:
    """
    Calculate distance of the path based on distances matrix

    Args:
        path: list
            Order of cities visited by the salesman
        distances: pd.DataFrame
            Matrix of distances between cities,
            cities numbers or id names as indices and columns
    """
    path_length = sum(distances.loc[x, y] for x, y in zip(path, path[1:]))
    # add distance back to the starting point
    path_length += distances.loc[path[0], path[-1]]
    return path_length


def solve_it(func: Callable):
    """
    Decorator for _solve method of TSPAlgorithm

    It must be called from inside of the algorithm
    as it takes TSPAlgorithm itself as the first parameter

    Developer note:
        decorate _solve method of TSPAlgorithm with solve_it
        and keep it in line with TSPAlgorithm iterface
        in order for it to return Result object

    Functionality:
        - Measures time of solving the problem (run time of _solve method)
        - Catches KeyboardInterrupt Exception to break algorithm,
          save and return current candidate solution
        - With help of ResultManager keeps track of all runs and
          saves Result in csv file

    Check out:

    src.utils.heuristic_algorithm TSPAlgorithm
    """

    def wrapper(algo, *args, **kwargs) -> Union[Result, None]:
        tic = time.perf_counter()
        try:
            result = func(algo, *args, **kwargs)
        # catches manual interuption of _solve method
        except KeyboardInterrupt:
            # if function is not able to access path_ or history
            # for some reason or it is empty - returns None
            if not hasattr(algo, 'path_'):
                return None
            if not hasattr(algo, "history"):
                return None
            if len(algo.history) == 0:
                return None

            # results object representing the state of the
            # algorithm at the moment of interuption
            result = Result(
                algorithm=algo,
                path=algo.path_,
                distance=algo.history[-1],
                distance_history=algo.history,
            )
            print(f"Algorithm has been stopped manually\n")
        toc = time.perf_counter()
        # assigns time to Result object
        result.time = toc - tic
        # save result in csv file
        ResultManager.save_result(result=result)
        return result

    return wrapper


class StopAlgorithm(Exception):
    """
    Used in IteratingAlgorithm as one of break conditions

    Developer note:
        Raise this Exception inside _run_iteration method
        to stop the iterative algorithm

    Check out for more information:

    src.utils.iterating_algorithm IteratingAlgorithm
    """

    def __init__(self, iteration: int, distance: int) -> None:
        self.message = (
            f"Algorithm stopped at {iteration} iteration\nBest distance: {distance}"
        )
        super().__init__(self.message)


def load_data(path: str, as_array: bool = False) -> Union[pd.DataFrame, np.array]:
    """
    Loads distances matrix data from given path
    File must have .xlsx extension for excel file
    Columns and indices of distances matrix must allign

    Args:
        path (str): Path to data
    Returns:
        data : pd.DataFrame or np.ndarray
    """

    data = pd.read_excel(path, index_col=0)
    # checks if format is correct
    distances_matrix_check(distances=data)
    return data.to_numpy() if as_array else data


def distances_matrix_check(distances: pd.DataFrame) -> None:
    """
    Checks if distances matrix is correct

    Params:
        distances: pd.DataFrame
            Matrix of distances between cities,
            cities numbers or id names as indices and columns

    The only check inplemented is whether indices of
    distances matrix allign with column names.
    """
    mes = "indices and columns of distances matrix should be equal"
    assert distances.index.to_list() == distances.columns.to_list(), mes


def path_check(path: list, distances: pd.DataFrame) -> None:
    """
    Runs the series of checks to assert path correctness for TSP problem

    Params:
        path: list
            Order of cities visited by the salesman
        distances: pd.DataFrame
            Matrix of distances between cities,
            cities numbers or id names as indices and columns

    Basic checks include:
        checking whether path is iterable
        checking whether path is of the same leghth as indices in distances matrix
        checking whether path alligns with distances matrix indices
        checking whether path has unique elements
    """
    assert isinstance(path, Iterable), "start_order must be iterable"
    assert len(path) == len(
        distances
    ), f"Expected {len(distances)} elements, got {len(path)}"
    assert all(
        index in distances.index.to_list() for index in path
    ), "elements of start_order must allign with distance matrix indices"
    assert len(set(path)) == len(path), "elements in start_order must be unique"


def get_random_path(indices: Union[list, pd.Index]) -> list:
    """
    Chooses random path from indices as the cities

    Args:
        indices: list | pd.Index
            Cities to be visited by the salesman
    """
    # indices of the cities
    if isinstance(indices, pd.Index):
        indices = list(indices)
    # return random path
    return random.sample(indices, len(indices))


class ResultManager:
    """
    Class for saving Result object into csv file

    Methods:
        save_result (staticmethod) - saves result into csv file
    """

    @staticmethod
    def save_result(result: Result, path: str = "results.csv") -> None:
        """
        Saves result into csv file

        Params:
            result: Result
                Result object to be saved in csv file
            path : str
                directory in which Result object will be saved
                with .csv extension, by default, results.csv

        If file doesn't exist
        """
        assert path.endswith(".csv"), "path must have .csv extension"
        res_df = result.to_df()
        res_df["run_time"] = datetime.now()
        # in new file header=True
        header = not os.path.exists(path)
        # creates directory if it doesn't exist
        ResultManager._create_directory(path=path)
        # append result to csv if exist, if not make new file
        res_df.to_csv(path, mode="a", index=False, header=header)

    @staticmethod
    def _create_directory(path: str) -> None:
        """
        Creates directory if it doesn't already exist

        Params:
            path: str
                path to the file

        Makes sure that the directory we will try to access exists
        """
        # if path exists
        if os.path.exists(path):
            return None

        split = path.split("/")
        # if non-existing directory was passed
        if len(split) > 1:
            # seperate folder from file
            folder_path = "/".join(split[:-1])
            # if folder doesn't exist, create directory
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
