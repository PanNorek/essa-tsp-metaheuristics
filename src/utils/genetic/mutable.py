from abc import ABC, abstractmethod
import numpy as np


class Mutable(ABC):
    """Abstract class for mutable"""

    @abstractmethod
    def mutate(self, mutation_rate: float) -> None:
        """Mutate individual.

        Args:
            mutation_rate (float): Probability of mutation
        """
        pass


class SimpleSwap(Mutable):
    """Simple inversion mutation"""

    def mutate(self, mutation_rate: float = 0) -> None:

        # decide if mutation will occur
        if np.random.choice((True, False), p=[mutation_rate, 1 - mutation_rate]):
            # select two random cities to swap
            swapped, swap_with = np.random.choice(len(self.path), 2, replace=False)
            # swap
            self.path[swapped], self.path[swap_with] = (
                self.path[swap_with],
                self.path[swapped],
            )


class Inversion(Mutable):
    """Inversion mutation"""

    def mutate(self, mutation_rate: float = 0) -> None:
        # decide if mutation will occur
        if np.random.choice((True, False), p=[mutation_rate, 1 - mutation_rate]):
            n = len(self.path)
            # select city to start and end inversion
            start = np.random.randint(0, n)
            end = np.random.randint(start, n)
            # invert
            self.path[start:end] = self.path[start:end][::-1]


class Insertion(Mutable):
    """Mutation by insertion"""

    def mutate(self, mutation_rate: float = 0) -> None:
        # decide if mutation will occur
        if np.random.choice((True, False), p=[mutation_rate, 1 - mutation_rate]):
            n = len(self.path)
            # select city to insert
            insert = np.random.randint(0, n)
            # select place to insert
            place = np.random.randint(0, n)
            # insert
            self.path.insert(place, self.path.pop(insert))
