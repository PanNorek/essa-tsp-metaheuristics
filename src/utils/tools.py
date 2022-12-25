import pandas as pd
import time
from typing import Callable, Union, Any
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from functools import wraps
import os


@dataclass
class Result:
    """Class for storing the results of the algorithm"""

    algorithm: Any
    path: list
    distance: int
    time: float = None
    distance_history: list = None
    mean_distance_history: list = None

    def __str__(self) -> str:
        mes = f"""distance: {self.distance}
                  algorithm: {self.algorithm}
                  solving time: {self.time:.3f} s
                """.replace("  ", "")
        if self.distance_history:
            mes += f"""history: {self.distance_history}\n"""
        return mes

    def __repr__(self) -> str:
        return str(self)

    def __gt__(self, object_) -> bool:
        assert isinstance(
            object_, Result
        ), f"can't compare result with {type(object_)} type"
        return self.distance > object_.distance

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data={
                "algorithm": [self.algorithm],
                "path": [self.path],
                "distance": [self.distance],
                "solving_time": [self.time],
            }
        )
        return df

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "path": self.path,
            "distance": self.distance,
            "solving_time": self.time,
        }


def get_path_distance(path: list, distances: pd.DataFrame) -> int:
    """Calculate distance of the path based on distances matrix"""
    path_length = sum(distances.loc[x, y] for x, y in zip(path, path[1:]))
    # add distance back to the starting point
    path_length += distances.loc[path[0], path[-1]]
    return path_length


def solve_it(func: Callable):
    """Decorator to measure time of solve method of Algorithm"""

    def wrapper(algo, *args, **kwargs):
        tic = time.perf_counter()
        try:
            result = func(algo, *args, **kwargs)
        except KeyboardInterrupt:
            result = Result(
                algorithm=algo,
                path=algo.best_path,
                distance=min(algo.history),
                distance_history=algo.history
            )
            print(f'Algorithm has been stopped manually\n')
        toc = time.perf_counter()
        result.time = toc - tic
        ResultManager.save_result(result=result)
        return result

    return wrapper


class StopAlgorithm(Exception):
    """Exception raised when algorithm stops."""

    def __init__(self, iteration: int, distance: int) -> None:
        self.message = (
            f"Algorithm stopped at {iteration} iteration\nBest distance: {distance}"
        )
        super().__init__(self.message)


def load_data(path: str,
              triu: bool = False,
              as_array: bool = False
              ) -> Union[pd.DataFrame, np.array]:
    """
    Load data from given path
    Args:
        path (str): Path to data
    Returns:
        data : pd.DataFrame or np.ndarray
    """
    data = pd.read_excel(path, index_col=0)
    if triu:
        # Get upper triangle of data
        return pd.np.triu(data.to_numpy())
    return data.to_numpy() if as_array else data


class ResultManager:
    @staticmethod
    def save_result(result: Result, file: str = "results.csv") -> None:
        res_df = result.to_df()
        res_df["run_time"] = datetime.now()
        header = not os.path.exists(file)
        res_df.to_csv(file, mode="a", index=False, header=header)
