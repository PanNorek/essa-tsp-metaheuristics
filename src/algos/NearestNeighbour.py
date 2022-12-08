from src.algos.Algorithm import Algorithm
from src.utils.time_it import time_it

@time_it 
class NearestNeighbour(Algorithm):
    " Nearest Neighbour Algorithm"
    def __init__(self, distance_matrix, start):
        super().__init__(distance_matrix, start)

    def solve(self):
        self.end = self.start
        self.unvisited = list(range(1, len(self.distance_matrix) + 1))
        self.unvisited.remove(self.start)
        self.distance = 0 
        self.path = str(self.start)

        while self.unvisited:

            current = self.end

            nearest_distance = self.distance_matrix.iloc[current-1, self.unvisited[0]-1]

            for neighbour in self.unvisited:
                
                try:
                    if self.distance_matrix.iloc[current-1, self.unvisited[neighbour]-1] < nearest_distance:
                        current = neighbour
                except IndexError:
                    pass
            
            try:
                self.unvisited.remove(current)
            except ValueError:
                return self.distance
            self.path += '-' + str(current)
            self.distance += nearest_distance
            self.end = current
        
    def get_path(self):
        return self.path

