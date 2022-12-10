from src.algos import SimulatedAnnealing, NearestNeighbour, TabuSearch, HillClimber
from src.utils import load_data
from src.algos import MultistartAlgorithm


if __name__ == '__main__':

    df = load_data('data\TSP_29.xlsx')
    NUM_STARTS = 10


    results1 = MultistartAlgorithm()(SimulatedAnnealing, df, n_starts=NUM_STARTS, temp=1000, alpha=.9, n_iter=100, verbose=False)
    results2 = MultistartAlgorithm()(TabuSearch, df, n_starts=NUM_STARTS, verbose=False, tabu_length=3 , n_iter=30)
    results3 = MultistartAlgorithm()(HillClimber, df, n_starts=NUM_STARTS, verbose=False, n_iter=25)

    