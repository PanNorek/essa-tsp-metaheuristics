from abc import ABC, abstractmethod
import pandas as pd 
import numpy as np
from typing import Union

class Algorithm(ABC):
    def __init__(self, distance_matrix: Union[pd.DataFrame, np.ndarray], start):
        self.distance_matrix = distance_matrix
        self.start = start

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_path(self):
        pass

