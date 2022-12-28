from abc import ABC, abstractmethod
from typing import List, Tuple, NewType, Union
from .individual import Individual
import random
import pandas as pd

Parents = NewType("Parents", Tuple[Individual, Individual])


class ParentSelection(ABC):
    """
    Abstract class for offspring selection
    """

    @staticmethod
    @abstractmethod
    def select(generation: List[Individual], size: Union[int, float] = 0.5) -> Parents:
        """
        Selects two individuals (parents) for crossover

        Params:
            population (Population): current population
        """
        pass

    @staticmethod
    def _get_size(generation: List[Individual], size: Union[int, float] = 0.5) -> int:
        if isinstance(size, float):
            size = int(len(generation) * size)
        return size


class Truncation(ParentSelection):
    """
    Truncation offspring selection
    """

    @staticmethod
    def select(generation: List[Individual], size: Union[int, float] = 0.5) -> Parents:
        # calculate the number of trucated individuals
        size = ParentSelection._get_size(generation=generation, size=size)
        # randomly select parents from the better part of the generation
        return random.sample(generation[:size], 2)


class Tournament(ParentSelection):
    """
    Tournament offspring selection
    """

    @staticmethod
    def select(generation: List[Individual], size: Union[int, float] = 0.5) -> Parents:
        # calculate the number of participants
        size = ParentSelection._get_size(generation=generation, size=size)
        # select parents from 2 random batches
        batch_1 = random.sample(generation, size)
        parent_1 = sorted(batch_1, reverse=True)[0]
        batch_2 = random.sample(generation, size)
        parent_2 = sorted(batch_2, reverse=True)[0]
        return parent_1, parent_2


class Roulette(ParentSelection):
    """
    Roulette offspring selection
    """

    DIST_FITNESS = "cum_fitness"

    @staticmethod
    def select(generation: List[Individual], size: None = None) -> Parents:
        # load individuals
        df = pd.DataFrame.from_dict([ind.to_dict() for ind in generation])
        # calculate cumulative fitness
        cum_fit = df[Individual.FITNESS].cumsum()
        # divide by total fitness
        df[Roulette.DIST_FITNESS] = cum_fit.div(df[Individual.FITNESS].sum())
        # select parents
        mask1 = df[Roulette.DIST_FITNESS] > random.random()
        mask2 = df[Roulette.DIST_FITNESS] > random.random()
        return df[mask1].iloc[0], df[mask2].iloc[0]
