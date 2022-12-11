from src.algos import SimulatedAnnealing, NearestNeighbour, TabuSearch, HillClimber
from src.utils import load_data, PathPlotter, BenchmarkPlotter, DistanceHistoryPlotter
from src.algos import MultistartAlgorithm


if __name__ == "__main__":

    df = load_data(r"data\TSP_29.xlsx")
    NUM_STARTS = 5

    algorithms = [SimulatedAnnealing, NearestNeighbour, TabuSearch, HillClimber]

    results = []
    results.append(min([NearestNeighbour().solve(df, start=x + 1) for x in range(29)]))
    results.append(
        MultistartAlgorithm()(
            SimulatedAnnealing,
            df,
            n_starts=NUM_STARTS,
            temp=1000,
            alpha=0.9,
            n_iter=100,
            verbose=False,
        )
    )
    results.append(
        MultistartAlgorithm()(
            TabuSearch, df, n_starts=NUM_STARTS, verbose=False, tabu_length=3, n_iter=30
        )
    )
    results.append(
        MultistartAlgorithm()(
            HillClimber, df, n_starts=NUM_STARTS, verbose=False, n_iter=25
        )
    )

    # Example of plotting the path
    pp = PathPlotter()
    pp.plot(results[0])

    # Example of plotting the benchmark
    bp = BenchmarkPlotter()
    bp.plot(results)

    # Example of plotting the benchmark with custom labels and palette
    bp.plot(
        results, labels=["RAZ", "DWA", "TRZY", "CZTERY"], palette="Purples_d"
    )  # Reds_d, Blues_d, Greens_d, Purples_d, Oranges_d, Greys_d

    dhp = DistanceHistoryPlotter()
    dhp.plot(
        results[1:]
    )  # results[1:] to exclude NearestNeighbour - it has no distance history
