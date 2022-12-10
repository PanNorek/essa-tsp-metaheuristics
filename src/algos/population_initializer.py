import numpy as np


class PopulationInitializer:
    """Class to initialize the population of the genetic algorithm."""

    def __init__(self, population_size, chromosome_size, min_value, max_value):
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.min_value = min_value
        self.max_value = max_value

    def initialize(self):
        population = []
        for i in range(self.population_size):
            chromosome = []
            for j in range(self.chromosome_size):
                chromosome.append(np.random.uniform(self.min_value, self.max_value))
            population.append(chromosome)
        return population
