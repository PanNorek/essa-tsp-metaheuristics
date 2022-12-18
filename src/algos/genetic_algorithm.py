from typing import Union, List, Callable
import pandas as pd
import numpy as np
from . import Algorithm
from ..utils.genetic import Population, SimpleSwap, Inversion, Insertion, Mutable, ParentSelection
from ..utils import ResultManager, Result, time_it
from ..utils.genetic import Elitism, Roulette, Tournament


class GeneticAlgorithm(Algorithm):
    """Genetic algorithm for solving TSP problem

    Parameters
    ----------
    init_pop_size : int, optional
        Initial population size, by default 50.
    pop_size : int, optional
        Population size, by default 100
    no_generations : int, optional
        Number of generations, by default 10
    selection_method : str, optional
        Parents selection method, user can choose between
        ["elitism", "roulette", "tournament"],
        by default "elitism".
    crossover_rate : float, optional
        Crossing-over probability rate, by default 1.0
    elite_size : Union[int, None], optional
        Elite size is only used when selection_method is "elitism",
        it means the number of best individuals that will be selected to crossing over,
        by default None(None means 1/5 of population size)
    tournament_size : Union[int, None], optional
        Tournament size is only used when selection_method is "tournament",
        it means the number of random individuals that will be selected to tournament,
        in which the best individual will be selected to crossing over,
        by default None(None means 2)
    mutation_rate : float, optional
        Probability of mutation , by default 0.5
    neigh_type : str, optional
        Mutation type, user can choose between
        ["simple", "inversion", "insertion"],
        by default "simple"
    random_state : int, optional
        Random state, by default None
    verbose : bool, optional
        Verbose, by default False



    Methods
    -------
    solve(distances: pd.DataFrame) -> Result
        Solve TSP problem by given hyperparameters.
    """

    NAME = "GENETIC ALGORITHM"
    PARAM_REFERENCE = {
        "elitism": Elitism,
        "roulette": Roulette,
        "tournament": Tournament,
        "simple": SimpleSwap,
        "inversion": Inversion,
        "insertion": Insertion,
    }

    def __init__(
        self,
        init_pop_size: int = 50,
        pop_size: int = 100,
        no_generations: int = 10,
        selection_method: str = "elitism",
        crossover_rate: float = 1.0,
        elite_size: Union[int, None] = None,
        tournament_size: Union[int, None] = None,
        mutation_rate: float = 0.5,
        neigh_type: str = "simple",
        random_state: int = None,
        verbose: bool = False,
    ):
        super().__init__(neigh_type=neigh_type, verbose=verbose)
        self._init_pop_size = init_pop_size
        self._pop_size = pop_size
        self.no_generations = no_generations
        self._selection_method = selection_method
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self.history = [np.inf]
        self._mean_distance = [np.inf]

        self._elite_size = int(pop_size * 0.2) if elite_size is None else elite_size
        self._tournament_size = int(pop_size * 0.1) if tournament_size is None else tournament_size

        assert 0 < init_pop_size < pop_size, "Initial population size must be in (0, init_pop_size)"
        assert (
            0 < crossover_rate <= 1
        ), "Crossing-over  must in (0,1>"  # not zero, because while loop will never end
        assert issubclass(
            GeneticAlgorithm.PARAM_REFERENCE[neigh_type], Mutable
        ), "Wrong mutation type."
        assert issubclass(
            GeneticAlgorithm.PARAM_REFERENCE[selection_method], ParentSelection
        ), "Wrong selection method."
        assert 0 <= mutation_rate <= 1, "Mutation rate must be between 0 and 1."

        self._set_random_seed(random_state)

    @time_it
    def solve(self, distances: pd.DataFrame) -> Result:

        # 1st stage: Create random population
        population = Population(self._init_pop_size, self._pop_size)
        population.generate_population(distances)
        print(f"Initial population: {population.population[0]}")

        # 2nd stage: Loop for each generation

        for generation in range(self.no_generations):

            # I: Crossover - make children
            population.crossover(
                distances,
                crossover_method=self.PARAM_REFERENCE[self._selection_method],
                crossover_rate=self._crossover_rate,
                elite_size=self._elite_size,
                tournament_size=self._tournament_size,
            )

            # II: Mutation - mutate all population
            population.mutate(
                distances,
                mutation_type=self.PARAM_REFERENCE[self.neigh_type],
                mutation_rate=self._mutation_rate,
            )

            # III: Selection - select the best half of the population
            population.select()

            if self._verbose:
                print(f"Generation {generation}: {population.population[0]}")

            # story only better results
            # if population.population[0].distance < self.history[-1]:
            #     self.history.append(population.population[0].distance)

            # or all results
            self.history.append(population.population[0].distance)

            self._mean_distance.append(population.calculate_mean_distance())

        print(f"Final population: {population.population[0]}")
        # write to file
        ResultManager.save_result(self.__dict__, distances.shape[0], population.population[0])
        # Return the best path

        return Result(
            algorithm=self,
            path=population.population[0].path,
            best_distance=self.history[-1],
            distance_history=self.history,
            mean_distance=self._mean_distance,
        )
