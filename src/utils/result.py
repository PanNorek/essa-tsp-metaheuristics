from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    algorithm: Any
    path: list
    best_distance: int
    time: float = None
    distance_history: list = None
    no_starts: int = None

    def __str__(self) -> str:
        mes = f"""best distance: {self.best_distance}
                  algorithm: {self.algorithm}
                  solving time: {self.time:.3f} s
                """.replace('  ', '')
        if self.distance_history:
            mes += f"""history: {self.distance_history}\n"""
        if self.no_starts:
            mes += f"""No. of starts: {self.no_starts}"""
        return mes

    def __repr__(self) -> str:
        return str(self)

    def __gt__(self, object) -> bool:
        assert isinstance(object, Result), f"can't compare result with {type(object)} type"
        return self.best_distance > object.best_distance
