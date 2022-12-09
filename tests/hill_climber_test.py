import unittest
from src.algos.hill_climber import HillClimber
from src.loaders.load_data import load_data
import numpy as np

class TestHillClimber(unittest.TestCase):
    def setUp(self):
        self.df = load_data('data/TSP_29.xlsx')
        self.hc = HillClimber()

    def test_compute_distance(self):
        self.assertEqual(self.hc._compute_distance(np.array(range(1,30)), self.df), 5752)

    def test_find_neighbours(self):
        self.assertEqual(len(self.hc._find_neighbours(list(range(1,30)))), 407)

    def test_swap(self):
        self.assertEqual(self.hc._swap(list(range(1,5)), 0, 1), [2, 1, 3, 4])
    
    def test_find_best_neighbour(self):
        sample = [list(range(1,30)), 
        [14, 18, 15,  4, 10, 20,  2, 21,  1, 28,  6, 12,  9,  5, 26, 29,  3, 13, 24, 27,  8, 16, 23, 25,  7, 19, 11, 22, 17]]
        self.assertEqual(self.hc._find_best_neighbour(sample, distance_matrix=self.df), [14, 18, 15,  4, 10, 20,  2, 21,  1, 28,  6, 12,  9,  5, 26, 29,  3, 13, 24, 27,  8, 16, 23, 25,  7, 19, 11, 22, 17])
    
    
