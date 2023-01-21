import random
import pandas as pd
from typing import Union, List
from ..tools import get_order_cost, get_random_path
from .individual import Individual
from ..neighbourhood_type import NeighbourhoodType
from .parent_selection import ParentSelection
from .crossover import CrossoverMethod


class Population:
    """
    Population of the individuals in genetic algorithm

    Methods:
        generate_population - generates a population of random individuals
            based on the distances matrix
        mutate - Mutates the chromosomes of individuals
            using neighbourhood type concept
        crossover - Creates a new generation by crossing over existing one

    Attributes:
        mean_distance - mean distance (cost function) in the population
        population - list of all individuals forming up the population
        best - best individual in the population

    Population consists of individuals (candidate solutions) that are evolved
    toward better solution

    Each candidate solution has a set of properties (its chromosomes or genotype)
    which can be mutated and altered.

    Mutate method mutates individuals of the population. After breeding the new
    generation, individuals are mutated. It helps not to get stuck in local minimum
    and diversify more the solutions in the population.

    Check out:

    src.utils.genetic.individual Individual

    src.algos.genetic_algorithm GeneticAlgorithm
    """

    def __init__(self, pop_size: int = 500) -> None:
        """
           Params:
            pop_size: int
                The number of individuals in the population
        """
        self._pop_size = pop_size
        # empty population is inicialized
        self._population = []

    def generate_population(self, distances: pd.DataFrame):
        """
        Generate a population of random individuals

        Params:
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order

        Returns:
            Population (self)
        """
        indices = distances.index.to_list()
        # get pop_size number of random paths
        paths = [get_random_path(indices=indices) for _ in range(self._pop_size)]
        # get distances of radom paths
        distances = [
            get_order_cost(order=path, cost_matrix=distances) for path in paths
        ]
        # create a list of individuals from random paths and assing it to _population attr
        self._population = [
            Individual(path=path, distance=distance)
            for path, distance in zip(paths, distances)
        ]
        return self

    @property
    def mean_distance(self) -> float:
        """Mean distance in the population"""
        return sum(ind.distance for ind in self._population) / len(self._population)

    @property
    def population(self) -> List[Individual]:
        """List of individuals constituting a population"""
        return self._population

    @property
    def best(self) -> Individual:
        """Best individual in a population (best candidate solution)"""
        return max(self._population)

    def _sort(self):
        """
        Sorts individuals in a population based on their
        fitness funtion in descending order
        """
        self._population.sort(reverse=True)

    def crossover(
        self,
        distances: pd.DataFrame,
        mating_pool_size: Union[int, float],
        selection_method: ParentSelection,
        crossover_method: CrossoverMethod,
        elite_size: int = 0,
    ) -> None:
        """
        Reproduces a population by crossing over individulas
        and fills up a new generation

        Params:
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order
            mating_pool_size: int | float
                Size of the sample from which parents will be selected
            selection_method: ParentSelection
                Parent selection method class used in the algorithm
            crossover_method: CrossoverMethod
                Crossover method object used in the algorithm
            elite_size: int | float
                The number of the elite individuals

        selection_method:
            - Truncation:
                random two individuals from x best individuals in population are selected
            - Tournament:
                the sample of x individuals is drawn randomly from the population
                and the best one is selcted for both parents
            - Roulette:
                Stochastic selection method, where the probability for selection
                of an individual is proportional to its fitness.

            Takes in an class ParentSelection and uses its static method select

        crossover_method:
            - PMX:
                Partially Mapped Crossover
                A way to combine two individuals resulting in two new children.
                - carve out randomly selected slice of each parent and replace it in its counterpart
                - ensure that the "unique item" invariant for both child individuals is maintained
            - OX:
                Ordered Crossover
                A variation of PMX with a different repairing procedure

            Takes in an object of type CrossoverMethod and uses its method crossover

        elite_size:
            number of the fittest individuals that goes directly to the next generation.

            If float:
                Value must be between 0 and 1 - the percentage of population considered an elite
            If int:
                Must not be greater then population - number of individuals considered an elite

        mating_pool_size:
            The size of sample from population from which parents will be drawn
            A way to filter out bad inferior individuals and select parents from the fittest
            Larger mating_pool_size gives more chance for bad individuals to breed
            Smaller mating_pool_size may cause homogenity in population

            If float:
                Value must be between 0 and 1 - the percentage of the fittest
                population from which parents will be drawn
            If int:
                Must not be greater then population - the number of the fittest
                individuals in population from which parents will be drawn
        """
        self._sort()
        # elite size goes directly to the next population
        new_population = self._population[:elite_size]
        # while the population is not full
        while len(new_population) < self._pop_size:
            # select parents with a given method
            parent_1, parent_2 = selection_method.select(
                population=self._population[elite_size:],
                size=mating_pool_size
            )
            # cross over with a given method
            # returns two lists representing new paths
            child_1, child_2 = crossover_method.crossover(
                parent_1=parent_1.path, parent_2=parent_2.path
            )
            # create Individual object from new solutions
            child_1, child_2 = (
                Individual(
                    path=child_1,
                    distance=get_order_cost(order=child_1, cost_matrix=distances),
                ),
                Individual(
                    path=child_2,
                    distance=get_order_cost(order=child_2, cost_matrix=distances),
                ),
            )
            # add first child to population
            new_population.append(child_1)
            # check if population is full
            if len(new_population) < self._pop_size:
                # add second child if not
                new_population.append(child_2)
        # check if length of new generation is correct
        assert (
            len(self._population) == self._pop_size
        ), f"Crossover failed Population should have {self._pop_size} \
            individuals, not {len(self._population)} "
        # assigns new generation to _population attr
        self._population = new_population

    def mutate(
        self,
        distances: pd.DataFrame,
        mutation: NeighbourhoodType,
        elite_size: Union[int, float] = 0,
        mutation_rate: float = 0.5,
    ) -> None:
        """
        Mutates individuals in population

        Params:
            distances: pd.DataFrame
                Matrix of set of jobs scheduled on a set of machines in a specific order
            mutation: NeighbourhoodType
                Type of neighbourhood used to mutate individual
            elite_size: int | float
                The number of the elite individuals
            mutation_rate: float
                Probability of mutation to occur

        mutation:
            Uses concept of neighbourhood to mutate new individuals
            The way not to get stuck in local minimum

            - Swap: swapping two elements in a list
            - Inversion: inversing order of a slice of a list
            - Insertion: inserting element into a place

        elite_size:
            number of the fittest individuals that goes directly to the next generation.

            If float:
                Value must be between 0 and 1 - the percentage of population considered an elite
            If int:
                Must not be greater then population - number of individuals considered an elite

        mutation_rate:
            The probability of individual to mutate
            Value must be between 0 and 1
        """
        # elite_size to float to slice the list
        if isinstance(elite_size, float):
            elite_size = int(len(self._population) * elite_size)

        # for each individual not in elite
        for individual in self._population[elite_size:]:
            # with mutation_rate probability
            if random.random() > mutation_rate:
                continue
            # mutate an individual
            individual.mutate(mutation=mutation)
            # and set its new distance
            individual.distance = get_order_cost(
                order=individual.path, cost_matrix=distances
            )

    # magic methods
    # --------------

    def __str__(self) -> str:
        """String representation of the population"""
        return f"Size: {len(self)}\nbest: {self.best}"

    def __repr__(self) -> str:
        """String representation of the population"""
        return str(self)

    def __len__(self):
        """The number of individuals in a population"""
        return len(self.population)

    def __getitem__(self, key):
        # population can be silced as a normal list
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self[i] for i in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index {key} is out of range")
            return self.population[key]
        else:
            raise TypeError("Invalid argument type")

    def __iter__(self):
        self.__i = 0
        return self

    def __next__(self):
        # population is iterable
        if self.__i >= len(self.population):
            raise StopIteration
        element = self.population[self.__i]
        self.__i += 1
        return element
