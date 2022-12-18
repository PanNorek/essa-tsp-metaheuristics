from abc import ABC, abstractmethod
from typing import List, Tuple
from .individual import Individual
import random
import pandas as pd


class ParentSelection(ABC):
    """
    Abstract class for offspring selection
    """

    @staticmethod
    @abstractmethod
    def select(generation: List[Individual], **kwargs):
        """
        Selects the next generation from the current population

        Args:
            population (Population): current population

        """
        pass


class Elitism(ParentSelection):
    """
    Elitism offspring selection
    """

    @staticmethod
    def select(generation: List[Individual], **kwargs) -> Tuple[Individual, Individual]:
        if "elite_size" in kwargs:
            return random.sample(generation[: kwargs.get("elite_size")], 2)
        else:
            raise ValueError("Elite size not specified.")


class Tournament(ParentSelection):
    """
    Tournament offspring selection
    """

    @staticmethod
    def select(generation: List[Individual], **kwargs) -> Tuple[Individual, Individual]:

        if "tournament_size" not in kwargs:
            raise ValueError("Tournament size not specified.")

        tournament_participants1 = random.sample(generation, kwargs.get("tournament_size"))
        tournament_participants1.sort(key=lambda x: x.fitness, reverse=True)
        tournament_participants2 = random.sample(generation, kwargs.get("tournament_size"))
        tournament_participants2.sort(key=lambda x: x.fitness, reverse=True)
        return tournament_participants1[0], tournament_participants2[0]


class Roulette(ParentSelection):
    """
    Roulette offspring selection
    """

    DIST_FITNESS = "cum_fitness"

    @staticmethod
    def select(generation: List[Individual], **kwargs) -> Tuple[Individual, Individual]:
        df = pd.DataFrame.from_records(vars(o) for o in generation)
        df[Individual.FITNESS] = (1 / df[Individual.DISTANCE]).astype("float64")
        cum_fit = df[Individual.FITNESS].cumsum()
        df[Roulette.DIST_FITNESS] = cum_fit.div(df[Individual.FITNESS].sum())
        mask1 = df[Roulette.DIST_FITNESS] > random.random()
        mask2 = df[Roulette.DIST_FITNESS] > random.random()
        return df[mask1].iloc[0], df[mask2].iloc[0]
