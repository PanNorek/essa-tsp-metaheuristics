from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Callable, Optional, Union
import numpy as np

class Mutable(ABC):
    """Abstract class for mutable"""

    @abstractmethod
    def mutate(self, mutation_rate: int) -> None:
        """Mutate individual.

        Args:
            mutation_rate (int): Probability of mutation
        """
        pass


class SimpleSwap(Mutable):
    """Simple inversion mutation"""

    def mutate(self, mutation_rate: float = 0) -> None:

        if np.random.choice((True, False), p=[mutation_rate, 1 - mutation_rate]):
            print("Mutating")
            swapped, swap_with = np.random.choice(len(self.path), 2, replace=False)
            self.path[swapped], self.path[swap_with] = self.path[swap_with], self.path[swapped]

class Inversion(Mutable):
    """Inversion mutation"""

    def mutate(self, mutation_rate: float = 0) -> None:
        if np.random.choice((True, False), p=[mutation_rate, 1 - mutation_rate]):
            n = len(self.path)
            start = np.random.randint(0, n)
            end = np.random.randint(start, n)
            self.path[start:end] = self.path[start:end][::-1]
            