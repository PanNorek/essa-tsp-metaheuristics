from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    """Class for storing the results of the algorithm"""

    algorithm: Any
    path: list
    best_distance: int
    time: float = None
    distance_history: list = None
    mean_distances: list = None

    def __str__(self) -> str:
        mes = f"""best distance: {self.best_distance}
                  algorithm: {self.algorithm}
                  solving time: {self.time:.3f} s
                """.replace(
            "  ", ""
        )
        if self.distance_history:
            mes += f"""history: {self.distance_history}\n"""
        return mes

    def __repr__(self) -> str:
        return str(self)

    def __gt__(self, object_) -> bool:
        assert isinstance(
            object_, Result
        ), f"can't compare result with {type(object_)} type"
        return self.best_distance > object_.best_distance
