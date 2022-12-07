from src.algos.Algorithm import Algorithm

def time_it(func):
    import time
    def wrapper(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        toc = time.time()
        print(f"Time elapsed: {toc-tic}")
        return result
    return wrapper

@time_it 
class NearestNeighbour(Algorithm):
    def __init__(self, distance_matrix, start):
        super().__init__(distance_matrix, start)

    def run(self):
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
            self.path += str(current)
            self.distance += nearest_distance
            self.end = current
        # return self.distance
    
    def get_path(self):
        return self.path

