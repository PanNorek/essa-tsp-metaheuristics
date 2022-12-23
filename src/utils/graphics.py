from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from .tools import Result


class PathPlotter:
    """Plot the path of the salesman"""

    def plot(self, result: Result):
        df = pd.DataFrame(
            zip(result.path, result.path[1:] + [result.path[0]]), columns=["from", "to"]
        )
        G = nx.from_pandas_edgelist(df, "from", "to", create_using=nx.DiGraph)
        plt.figure(figsize=(10, 6))
        plt.title(f"Salesman Path, Distance: {result.best_distance}")
        nx.draw(G, with_labels=True)
        plt.show()


class BenchmarkPlotter:
    """Plot the benchmark of the algorithms"""

    def plot(
        self, results: List[Result], labels: List[str] = None, palette: str = "Blues_d"
    ):
        """Plot the benchmark of the algorithms"""
        plt.figure(figsize=(10, 6))
        if labels is None:
            labels = [result.algorithm.NAME for result in results]
        ax = sns.barplot(
            x=labels, y=[result.best_distance for result in results], palette=palette
        )
        for p in ax.containers:
            ax.bar_label(p, label_type="edge")

        plt.title("Algorithms Benchmark")
        plt.xlabel("Algorithm")
        plt.ylabel("Distance")
        plt.show()


class DistanceHistoryPlotter:
    """Plot the distance history of the algorithm"""

    def plot(self, results: List[Result], labels: List[str] = None):

        if all(result.algorithm.NAME == "GENETIC ALGORITHM" for result in results):
            plt.figure(figsize=(10, 6))
            # fig, axs = plt.subplots(2, 1, figsize=(10, 12))

            if labels is None:
                labels = [f"Algorithm {i}" for i in range(1, len(results) + 1)]

            for result, label in zip(results, labels):
                plt.plot(result.distance_history, label=label + " - best distance")
                plt.plot(
                    result.mean_distance_history,
                    label=label + " - mean distance",
                    linestyle="--",
                )
            # for result, label in zip(results, labels):
            #     axs[0].plot(result.distance_history, label=label)
            #     axs[1].plot(result.mean_distance, label=label)

            # axs[0].set_title("Distance History")
            plt.title("Distance History")
            # axs[0].set_xlabel("Iteration")
            plt.xlabel("Iteration")
            # axs[0].set_ylabel("Distance")
            plt.ylabel("Distance")
            # axs[0].legend(loc="best")
            plt.legend(loc="best")

            # axs[1].set_title("Mean Distance History")
            # axs[1].set_xlabel("Iteration")
            # axs[1].set_ylabel("Distance")
            # axs[1].legend(loc="best")

            plt.show()
            return
        else:
            plt.figure(figsize=(10, 6))
            if labels is None:
                labels = list(range(len(results)))

            for result, label in zip(results, labels):
                plt.plot(result.distance_history, label=label)

            plt.title("Distance History")
            plt.xlabel("Iteration")
            plt.ylabel("Distance")
            # show legend in the best location
            plt.legend(loc="best")
            plt.show()
