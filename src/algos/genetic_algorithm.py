# from typing import Union, List, Callable
# import pandas as pd
# import numpy as np
from . import Algorithm
# # from ..utils.genetic import Population, MutationSwap, MutationInversion, MutationInsertion, NeighbourhoodType
# from ..utils import ResultManager, Result
# from ..utils.genetic import Elitism, Roulette, Tournament
# import time


class GeneticAlgorithm(Algorithm):
    pass
#     """Genetic algorithm for solving TSP problem"""

#     NAME = "GENETIC ALGORITHM"
#     PARAM_REFERENCE = {
#         "elitism": Elitism,
#         "roulette": Roulette,
#         "tournament": Tournament,
#         "simple": MutationSwap,
#         "inversion": MutationInversion,
#         "insertion": MutationInsertion,
#     }

#     def __init__(
#         self,
#         init_pop_size: int = 50,
#         pop_size: int = 100,
#         no_generations: int = 10,
#         selection_method: str = "elitism",
#         crossover_rate: float = 1.0,
#         elite_size: Union[int, None] = None,
#         tournament_size: Union[int, None] = None,
#         mutation_rate: float = 0.5,
#         neigh_type: str = "simple",
#         random_state: int = None,
#         verbose: bool = False,
#     ):
#         super().__init__(neigh_type=neigh_type, verbose=verbose)
#         self._init_pop_size = init_pop_size
#         self._pop_size = pop_size
#         self.no_generations = no_generations
#         self._selection_method = selection_method
#         self._crossover_rate = crossover_rate
#         self._mutation_rate = mutation_rate
#         self._history = [np.inf]
#         self._mean_distance = []

#         self._elite_size = int(pop_size * 0.2) if elite_size is None else elite_size
#         self._tournament_size = int(pop_size * 0.1) if tournament_size is None else tournament_size

#         assert 0 < init_pop_size < pop_size, "Initial population size must be in (0, init_pop_size)"
#         assert (
#             0 < crossover_rate <= 1
#         ), "Crossing-over  must in (0,1>"  # not zero, because while loop will never end
#         assert issubclass(
#             GeneticAlgorithm.PARAM_REFERENCE[neigh_type], NeighbourhoodType
#         ), "Wrong mutation type."
#         assert 0 <= mutation_rate <= 1, "Mutation rate must be between 0 and 1."

#         self._set_random_seed(random_state)

#     def solve(self, distances: pd.DataFrame) -> pd.DataFrame:
#         start = time.perf_counter()

#         # 1st stage: Create random population
#         population = Population(self._init_pop_size, self._pop_size)
#         population.generate_population(distances)
#         print(f"Initial population: {population.population[0]}")

#         # 2nd stage: Loop for each generation

#         for generation in range(self.no_generations):
#             tic = time.perf_counter()

#             # I: Crossover - make children
#             population.crossover(
#                 distances,
#                 crossover_method=self.PARAM_REFERENCE[self._selection_method],
#                 crossover_rate=self._crossover_rate,
#                 elite_size=self._elite_size,
#                 tournament_size=self._tournament_size,
#             )

#             # II: Mutation - mutate all population
#             population.mutate(
#                 distances,
#                 mutation_type=self.PARAM_REFERENCE[self.neigh_type],
#                 mutation_rate=self._mutation_rate,
#             )

#             # III: Selection - select the best half of the population
#             population.select()

#             toc = time.perf_counter()
#             if self._verbose:
#                 print(
#                     f"Generation {generation}: {population.population[0]} took {toc - tic:0.4f} seconds"
#                 )
#             # story only better results
#             if population.population[0].distance < self._history[-1]:
#                 self._history.append(population.population[0].distance)

#             # self._mean_distance.append(population.mean_distance)
#         self._time = time.perf_counter() - start
#         print(f"Final population: {population.population[0]}")
#         # write to file
#         ResultManager.save_result(self.__dict__, distances.shape[0], population.population[0])
#         # Return the best path

#         return Result(
#             self.NAME,
#             population.population[0].path,
#             population.population[0].distance,
#             self._time,
#             self._history,
#         )
