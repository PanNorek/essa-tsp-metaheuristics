from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple


class Algorithm(ABC):
    # TODO: what of it should go to solve/run method?
    def __init__(self,
                 neigh_type: str,
                 ) -> None:
        self.neigh_type = neigh_type

    @abstractmethod
    def solve(self, distance_matrix: pd.DataFrame, start: int) -> Tuple[int, str]:
        pass

