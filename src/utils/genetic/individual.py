from typing import Union
from ..neighbourhood_type import NeighbourhoodType


class Individual:
    """
    One candidate solution in genetic algorithm

    Methods:
        mutate - Mutates a chromosome of individual
            using neighbourhood type concept
        to_dict - Transforms individual into a dictionary

    Attributes:
        path - Order of cities visited by the salesman (chromosome of the individual)
        distance - Distance to be traversed by the salesman with current path
        fitness - Value of the fitness function, the greater the value the better solution
            formula (1 / distance) is used to transform cost function into fitness function

    Also sometimes refered to as:
    creatures, organisms or phenotypes

    In a genetic algorithm, a population of candidate solutions
    is evolved toward better solutions.

    Each candidate solution has a set of properties (its chromosomes or genotype)
    which can be mutated and altered.
    Mutate method of individual is used to change the order of the path.

    Chromosome is a set of parameters which define a proposed solution
    to the problem that the genetic algorithm is trying to solve

    In TSP chromosome is represented by the path (order of the cities)
    One of the hindrances is a requirement of uniqueness. Genes must not
    repeat in the chromosome.

    Check out:

    src.utils.genetic.neighbourhood_type NeighbourhoodType

    src.utils.genetic.population Population

    src.algos.genetic_algorithm GeneticAlgorithm
    """

    # DataFrame columns
    PATH = "path"
    DISTANCE = "distance"
    INDIVIDUAL = "individual"
    FITNESS = "fitness"

    def __init__(self, path: list, distance: Union[int, float]) -> None:
        """
        Params:
            path: list
                Order of cities visited by the salesman
            distances: pd.DataFrame
                Matrix of distances between cities,
                cities numbers or id names as indices and columns
        """
        # what defines the solution
        # path - its chromosome
        # distance - cost function of the solution
        self.path = path
        self.distance = distance

    @property
    def fitness(self) -> float:
        """Returns value of the fitness function for individual"""
        return 1 / self.distance

    def mutate(self, mutation: NeighbourhoodType) -> None:
        """
        Mutates a chromosome of individual using neighbourhood type concept

        Params:
            mutation: NeighbourhoodType
                Type of neighbourhood used to mutate individual

        Sets random adjacent solution as the new chromosome.
        New distance must be set from outside of the class
        """
        self.path = mutation.switch(path=self.path, how="random")

    def to_dict(self) -> dict:
        """Dictionary representation of individual"""
        return {
            self.INDIVIDUAL: self,
            self.PATH: self.path,
            self.DISTANCE: self.distance,
            self.FITNESS: self.fitness,
        }

    def __assert_type(self, object_) -> None:
        """Throws an exception when  object is not of Individual type"""
        assert isinstance(
            object_, Individual
        ), f"Cannot compare with {type(object_)} type"

    def __gt__(self, object) -> bool:
        """Checks if compared individual is better fit based on their fitness function"""
        self.__assert_type(object)
        return self.fitness > object.fitness

    def __str__(self) -> str:
        """String representation of individual"""
        return f"Individual(path: {self.path[0]}...{self.path[-1]}, distance: {self.distance:.2f})"

    def __repr__(self) -> str:
        """String representation of individual"""
        return str(self)

    def __len__(self) -> int:
        """Length of individual is a length of its chromosome"""
        return len(self.path)
