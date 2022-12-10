from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    algorithm: Any
    path: list
    best_distance: int
    solving_time: float = None
    distance_history: list = None

    def __str__(self) -> str:
        return f"""best distance: {self.best_distance}
                   algorithm: {self.algorithm}
                   solving time: {self.solving_time:.3f} s
                   history: {self.distance_history}
                """.replace('  ', '')

    def __repr__(self) -> str:
        return str(self)

    def __gt__(self, object) -> bool:
        assert isinstance(object, Result), f"can't compare result with {type(object)} type"
        return self.best_distance > object.best_distance
