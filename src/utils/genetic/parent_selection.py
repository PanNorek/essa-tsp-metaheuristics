from abc import ABC, abstractmethod
from typing import List, Tuple, NewType, Union
from .individual import Individual
import random
import pandas as pd

Parents = NewType("Parents", Tuple[Individual, Individual])


class ParentSelection(ABC):
    """
    Interface for parent selection in genetic algorithm

    Methods:
        select - (static method) selects two individual
            from a list of individuals

    Selection is the stage of a genetic algorithm in which individual
    genomes are chosen from a population for later breeding.

    Three ways of parent selection are implemented:
        - Truncation:
            random two individuals from x best individuals in population are selected
        - Tournament:
            the sample of x individuals is drawn randomly from the population
            and the best one is selcted for both parents
        - Roulette:
            Stochastic selection method, where the probability for selection
            of an individual is proportional to its fitness.

    Check out:

    src.algos.genetic_algorithm GeneticAlgorithm
    """
    NAME = None

    @staticmethod
    @abstractmethod
    def select(population: List[Individual], size: Union[int, float] = 0.5) -> Parents:
        """
        Selects two individuals (parents) for crossover

        Params:
            population: list[Individuals]
                population of individuals in Genetic Algorithm
            size: int | float
                Number of best individuals forming up mating pool
                from which parents will be selected.

        Returns:
            Two selected individuals (parents)

        Check out:

        src.utils.genetic.individual Individual

        src.utils.genetic.population Population
        """
        pass

    @staticmethod
    def _get_size(population: List[Individual], size: Union[int, float] = 0.5) -> int:
        """
        Returns the number of best individuals considered
        for parent selection.

        Params:
            population: list[Individuals]
                population of individuals in Genetic Algorithm
            size: int | float
                The number of best individuals or percentage
                of best individuals considered for parent selection

        Returns:
            The number of best individuals considered for parent selection
            Mating pool size

        Check out:

        src.utils.genetic.individual Individual

        src.utils.genetic.population Population
        """
        if isinstance(size, float):
            size = int(len(population) * size)
        return size

    @staticmethod
    def _sort(population: List[Individual]) -> List[Individual]:
        """
        Sorts individuals in a population based on their
        fitness funtion in descending order
        """
        return sorted(population, reverse=True)


class Truncation(ParentSelection):
    """
    Truncation method for parent selection in genetic algorithm

    Methods:
        select - (static method) selects two individual
            from a list of individuals

    Random two individuals from x best individuals in population are selected.
    Only the best x individuals can be selected as parents.

    Check out:

    src.algos.genetic_algorithm GeneticAlgorithm
    """
    NAME = "Truncation"

    @staticmethod
    def select(population: List[Individual], size: Union[int, float] = 0.5) -> Parents:
        # calculate the number of individuals forming up mating pool
        size = ParentSelection._get_size(population=population, size=size)
        # sort population based on fitness function
        population = ParentSelection._sort(population=population)
        # randomly select parents from the better part of the generation
        return random.sample(population[:size], 2)


class Tournament(ParentSelection):
    """
    Tournament method for parent selection in genetic algorithm

    Methods:
        select - (static method) selects two individual
            from a list of individuals

    The sample of x individuals is drawn randomly from the population
    and the best one is selected for both parents

    Check out:

    src.algos.genetic_algorithm GeneticAlgorithm
    """
    NAME = "Tournament"

    @staticmethod
    def select(population: List[Individual], size: Union[int, float] = 0.5) -> Parents:
        # calculate the number of participants
        size = ParentSelection._get_size(population=population, size=size)
        # select parents from 2 random batches
        batch_1 = random.sample(population, size)
        # the best individual from random batch
        parent_1 = sorted(batch_1, reverse=True)[0]
        # action is repeated for second parent
        batch_2 = random.sample(population, size)
        parent_2 = sorted(batch_2, reverse=True)[0]
        # two parents (individuals) are returned
        return (parent_1, parent_2)


class Roulette(ParentSelection):
    """
    Roulette method for parent selection in genetic algorithm

    Methods:
        select - (static method) selects two individual
            from a list of individuals

    Stochastic selection method, where the probability for selection
    of an individual is proportional to its fitness.

    1. The fitness function is evaluated for each individual,
       providing fitness values, which are then normalized.
    2. Normalization means dividing the fitness value of each individual
       by the sum of all fitness values,
       so that the sum of all resulting fitness values equals 1.
    3. Accumulated normalized fitness values are computed:
       the accumulated fitness value of an individual is the sum
       of its own fitness value plus the fitness values of all the previous individuals;
       the accumulated fitness of the last individual should be 1,
       otherwise something went wrong in the normalization step.
    4. A random number R between 0 and 1 is chosen.
       The selected individual is the first one whose accumulated
       normalized value is greater than or equal to R.

    Check out:
    https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)

    src.algos.genetic_algorithm GeneticAlgorithm
    """
    NAME = "Roulette"
    DIST_FITNESS = "cum_fitness"

    @staticmethod
    def select(population: List[Individual], size: None = None) -> Parents:
        # load individuals
        df = pd.DataFrame.from_dict([ind.to_dict() for ind in population])
        # calculate cumulative fitness
        cum_fit = df[Individual.FITNESS].cumsum()
        # divide by total fitness
        # individuals with better fitness will have greater values,
        # hence greater probablity of being selected
        df[Roulette.DIST_FITNESS] = cum_fit.div(df[Individual.FITNESS].sum())
        # select parents
        mask1 = df[Roulette.DIST_FITNESS] > random.random()
        mask2 = df[Roulette.DIST_FITNESS] > random.random()
        # two parents (individuals) are returned
        return (
            df.loc[mask1, Individual.INDIVIDUAL].iloc[0],
            df.loc[mask2, Individual.INDIVIDUAL].iloc[0]
        )
