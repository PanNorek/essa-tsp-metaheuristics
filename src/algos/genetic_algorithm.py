from typing import Union
import pandas as pd
from .iterating_algorithm import IteratingAlgorithm
from ..utils.genetic import (
    Population,
    Truncation,
    Roulette,
    Tournament,
    PMX,
    OX,
)
from ..utils import Result, solve_it
import copy


class GeneticAlgorithm(IteratingAlgorithm):
    """
    Genetic algorithm for solving TSP problem

    Methods:
        solve - used for solving TSP problem

    Attributes:
        path_ - best path found by algorithm
        history - list of accepted solutions during iteration process
        population - list of all individuals forming up the population

    Metaheuristic inspired by the process of natural selection
    that belongs to the larger class of evolutionary algorithms (EA).
    Genetic algorithms are commonly used to generate high-quality
    solutions to optimization and search problems by relying on
    biologically inspired operators such as mutation, crossover and selection.

    In a genetic algorithm, a population of candidate solutions
    (in GA called Individuals) is evolved toward better solutions.
    Each candidate solution has a set of properties (its chromosomes or genotype)
    which can be mutated and altered.

    Check out:

    src.utils.genetic.individual Individual

    src.utils.genetic.population Population

    Chromosome is a set of parameters which define a proposed solution
    to the problem that the genetic algorithm is trying to solve

    In TSP chromosome is represented by the path (order of the cities)
    One of the hindrances is a requirement of uniqueness. Genes must not
    repeat in the chromosome. TSP problems use special methods of crossover
    to take care of this:

    Crossover, also called recombination, is a genetic operator used to
    combine the genetic information of two parents to generate new offspring.
    It is one way to stochastically generate new solutions from an existing population

    Two way of crossover are implemented:
        - PMX:
            A way to combine two individuals chromosome resulting in two new child chromosome.
            - carve out randomly selected slice of each parent and replace it in its counterpart
            - ensure that the "unique item" invariant for both child individuals is maintained
        - OX:
            A variation of PMX with a different repairing procedure

        For more information check out:

        src.utils.genetic.crossover CrossoverMethod, PMX, OX

    The evolution usually starts from a population of randomly generated individuals,
    and is an iterative process, with the population in each iteration called a generation.
    Here generation is just a list of individuals that is a part of
    Population object that is evolving with each generation

    Class Population has a method generate_population that is used to set up
    a random set of individuals

    In each generation, the fitness of every individual in the population is evaluated.

    In TSP distance is a cost function, in order to turn it into a fitness function
    following formula is applied:

    fitness = 1/distance

    The more fit individuals are stochastically selected from the current population

    Three ways of parent selection are implemented:
        - Truncation:
            random two individuals from x best individuals in population are selected
        - Tournament:
            the sample of x individuals is drawn randomly from the population
            and the best one is selected for both parents
        - Roulette:
            Stochastic selection method, where the probability for selection
            of an individual is proportional to its fitness.

    Selected parents are crossing over and breeding new individuals
    that constitute the new generation

    Then each individual's genome is modified to form a new generation.

    There are three ways of mutating which are based on NeighbourhoodType:
        - Swap:
            Two random genes are swapped
        - Insertion:
            One gene is inserted between different genes
        - Inversion
            A slice of the chromosome is inversed

    The way to keep the best individuals and push them without any
    modification to the next generation is called Elitism.

    In GeneticAlgorithm this concept is implemented by elite_size parameter
    which by default is 0. X number of the fittest individuals will go
    directly to the next generation

    The new generation of candidate solutions is then used in the next iteration of the algorithm.
    Algorithm terminates when a maximum number of generations has been produced

    For more information check out:
    https://en.wikipedia.org/wiki/Genetic_algorithm
    https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)
    https://en.wikipedia.org/wiki/Chromosome_(genetic_algorithm)
    https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
    https://mat.uab.cat/~alseda/MasterOpt/GeneticOperations.pdf
    """

    NAME = "GENETIC ALGORITHM"
    DEFAULT_ITERS = 10
    _SELECTION_METHODS = {
        "truncation": Truncation,
        "roulette": Roulette,
        "tournament": Tournament,
    }
    _CROSSOVER_METHODS = {"pmx": PMX, "ox": OX}

    def __init__(
        self,
        pop_size: int = 100,
        n_iter: int = DEFAULT_ITERS,
        selection_method: str = "tournament",
        crossover_method: str = "pmx",
        elite_size: Union[int, float] = 0,
        mating_pool_size: Union[int, float] = 0.5,
        mutation_rate: float = 0.5,
        mutation_type: str = "swap",
        verbose: bool = False,
    ) -> None:
        """
        Params:
            pop_size: int
                The number of individuals in the population
            n_iters: int
                Number of iterations to be run
            selection_method: str
                Parent selection method used in the algorithm
            crossover_method: str
                Crossover method used in the algorithm
            elite_size: int | float
                The number of the elite individuals
            mating_pool_size: int | float
                Size of the sample from which parents will be selected
            mutation_rate: float
                Probability of mutation to occur
            mutation_type: str
                Mutation type used in the algorithm
            verbose: bool
                If True, prints out information about algorithm progress

        selection_method:
            by deafault "truncation"
            - "truncation":
                random two individuals from x best individuals in population are selected
            - "tournament":
                the sample of x individuals is drawn randomly from the population
                and the best one is selcted for both parents
            - "roulette":
                Stochastic selection method, where the probability for selection
                of an individual is proportional to its fitness.

        crossover_method:
            by deafault "pmx"
            - "pmx":
                Partially Mapped Crossover
                A way to combine two individuals resulting in two new children.
                - carve out randomly selected slice of each parent and replace it in its counterpart
                - ensure that the "unique item" invariant for both child individuals is maintained
            - "ox":
                Ordered Crossover
                A variation of PMX with a different repairing procedure

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

        mutation_rate:
            The probability of individual to mutate
            Value must be between 0 and 1

        mutation_type:
            Uses concept of neighbourhood to mutate new individuals
            The way not to get stuck in local minimum

            - "swap": swapping two elements in a list
            - "inversion": inversing order of a slice of a list
            - "insertion": inserting element into a place
        """
        # IteratingAlgorithm constructor
        super().__init__(neigh_type=mutation_type, verbose=verbose, n_iter=n_iter)
        self._pop_size = pop_size
        # random population is drawn in solve method
        self.population_ = None
        self._selection_method = selection_method
        self._crossover_method = crossover_method
        self._mutation_rate = mutation_rate
        self._elite_size = elite_size
        self._mating_pool_size = mating_pool_size
        # list of mean distances in each generation
        self.mean_distances = []
        # checks if passed parameters are correct
        self.__check_params()

    def __check_params(self) -> None:
        """Checks if passed init params are correct"""
        # correct value for mutation rate
        assert 0 <= self._mutation_rate <= 1, "Mutation rate must be between 0 and 1."
        # correct value for selection method
        assert (
            self._selection_method in self._SELECTION_METHODS
        ), f"selection method must be one of {self._SELECTION_METHODS}"
        # correct value for crossover method
        assert (
            self._crossover_method in self._CROSSOVER_METHODS
        ), f"crossover method must be one of {self._CROSSOVER_METHODS}"
        # correct value type for elite size
        assert isinstance(
            self._elite_size, (int, float)
        ), "elite size must be int or float"
        # correct value for elite size if int
        if type(self._elite_size) == int:
            assert (
                0 <= self._elite_size <= self._pop_size
            ), "elite size must not be greater then population size"
        # correct value for elite size if float
        if type(self._elite_size) == float:
            assert (
                0 <= self._elite_size <= 1
            ), "elite size must be between 0 and 1 if passed as float"
            # sets to int
            self._elite_size = int(self._pop_size * self._elite_size)
        # correct value type for mating pool size
        assert isinstance(
            self._mating_pool_size, (int, float)
        ), "mating pool must be int or float"
        # correct value for mating pool size if int
        if type(self._mating_pool_size) == int:
            assert (
                0 <= self._mating_pool_size <= self._pop_size
            ), "elite size must not be greater then population size"
        # correct value for mating pool size if float
        if type(self._mating_pool_size) == float:
            assert (
                0 <= self._mating_pool_size <= 1
            ), "mating pool must be between 0 and 1 if passed as float"
            # sets to int - substract elite_size
            self._mating_pool_size = int(
                (self._pop_size - self._elite_size) * self._mating_pool_size
            )
        assert (
           (self._elite_size + self._mating_pool_size) <= self._pop_size
        ), "elite_size and mating_pole_size together exceed population size"

        # setting crossover object
        self._crossover = self._CROSSOVER_METHODS[self._crossover_method]()
        # setting selection class
        self._selection = self._SELECTION_METHODS[self._selection_method]

    @solve_it
    def _solve(
        self,
        distances: pd.DataFrame,
        random_seed: Union[int, None] = None,
        start_order: Population = None,
    ) -> pd.DataFrame:
        # method enable starting algorithm with specified population
        # it needs to be a Population object with non-empty list of invdividuals

        # 1st stage: Create random population
        self.population_: Population = self._setup_start(
            distances=distances, random_seed=random_seed, start_order=start_order
        )
        # 2nd stage: Loop for each generation
        for _ in range(self._n_iter):
            self._run_iteration(distances=distances)

        # returns Result object
        result = Result(
            algorithm=self,
            path=self.path_,
            distance=self._history[-1],
            distance_history=self._history,
            mean_distance_history=self.mean_distances,
        )
        return result

    def _run_iteration(self, distances: pd.DataFrame) -> None:
        # number of iteration is increased by one
        self._next_iter()
        # I: Parent Selection and Crossover - populate new generation
        self.population_.crossover(
            distances=distances,
            # mating pool size to draw parents from
            mating_pool_size=self._mating_pool_size,
            # class of selection method
            selection_method=self._selection,
            # crossover method object
            crossover_method=self._crossover,
            # elite size goes directly into the next generation
            elite_size=self._elite_size,
        )
        # II: Mutation - mutate entire population
        self.population_.mutate(
            distances=distances,
            # neighbourhood type object passed as mutation
            mutation=self._neigh,
            # elite size is skipped - not mutated
            elite_size=self._elite_size,
            # probability of mutation
            mutation_rate=self._mutation_rate,
        )
        self._history.append(self.population_.best.distance)
        # best distance from population is new candidate solution
        # to be in line with TSPAlgorithm interface
        self._path = self.population_.best.path
        # mean distances of the generation are stored in mean_distances attribute
        self.mean_distances.append(self.population_.mean_distance)
        # prints out info about new generation in verbose mode
        if self._verbose:
            print(
                f"Generation {self._i}\nBest distance: {self.population_.best.distance:.2f}"
            )
            print(
                f"Mean distance: {self.population_.mean_distance:.2f}"
            )

    def _set_start_order(
        self, distances: pd.DataFrame, start_order: Population = None
    ) -> Population:
        if start_order is not None:
            self._start_order_check(start_order=start_order)
            copy_order = copy.deepcopy(start_order)
            return copy_order
        # if start_order not specified
        # randomly generate population of individuals
        population = Population(pop_size=self._pop_size)
        population.generate_population(distances=distances)
        return population

    def _start_order_check(self, start_order: Population) -> None:
        # only Population object is accepted
        assert isinstance(start_order, Population), "start_order must be a Population object"
        # passed population must be of the same length as self._pop_size
        assert (
            len(start_order) == self._pop_size
        ), f"Expected population with {self._pop_size} elements, got {len(start_order)}"

    def __str__(self) -> str:
        mes = super().__str__()
        mes += f"""pop_size: {self._pop_size}\n\
        generations: {self._n_iter}\n\
        selection_method: {self._selection_method}\n\
        crossover_method: {self._crossover_method}\n\
        elite_size: {self._elite_size}\n\
        mating_pool_size: {self._mating_pool_size}\n\
        mutation_rate: {self._mutation_rate}\n"""
        return mes
