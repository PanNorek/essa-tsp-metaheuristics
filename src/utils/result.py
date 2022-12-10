from dataclasses import dataclass
from typing import Any


class Result(dataclass):
    algorithm: Any
    path: list
    best_distance: int
    distance_history: list
    solving_time: float

    def __str__(self) -> str:
        return f"""best distance: {self.best_distance}
                   algorithm: {self.algorithm}
                   solving time: {self.solving_time}
                """

    def __repr__(self) -> str:
        return str(self)

    def __gt__(self, object) -> bool:
        assert isinstance(object, Result), f"can't compare result with {type(object)} type"
        return self.best_distance > object.best_distance
