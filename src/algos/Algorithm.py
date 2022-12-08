from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union


class Algorithm(ABC):
    # TODO: what of it should go to solve/run method?
    def __init__(self,
                 distance_matrix: Union[pd.DataFrame, np.ndarray],
                 start: int,
                 neigh_type: str = None
                 ) -> None:
        self.distance_matrix = distance_matrix
        self.start = start
        self.neigh_type = neigh_type

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def get_path(self):
        pass
