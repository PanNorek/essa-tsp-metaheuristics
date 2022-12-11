import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from . import Result
from typing import List

class PathPlotter:
    """Plot the path of the salesman"""

    def plot(self, result: Result):
        df = pd.DataFrame(zip(result.path, result.path[1:] + [result.path[0]]), columns=['from', 'to'])
        G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph)

        plt.figure(figsize=(10, 6))
        plt.title(f"Salesman Path, Distance: {result.best_distance}")
        nx.draw(G, with_labels=True)
        plt.show()

class BenchmarkPlotter:
    """Plot the benchmark of the algorithms"""

    def plot(self, results: List[Result], labels: List[str] = None, palette: str="Blues_d"):
        """Plot the benchmark of the algorithms"""
        plt.figure(figsize=(10, 6))
        if labels is None:
            labels = [result.algorithm.NAME for result in results]
        ax = sns.barplot(x=labels, y=[result.best_distance for result in results], palette=palette)
        for p in ax.containers:
            ax.bar_label(p, label_type='edge')

        plt.title(f"Algorithms Benchmark")
        plt.xlabel("Algorithm")
        plt.ylabel("Distance")
        plt.show()

class DistanceHistoryPlotter:
    """Plot the distance history of the algorithm"""

    def plot(self, results: List[Result], labels: List[str] = None):
        plt.figure(figsize=(10, 6))
        if labels is None:
            labels = [result.algorithm.NAME for result in results]
        for result, label in zip(results, labels):
            plt.plot(result.distance_history, label=label)
        plt.title(f"Distance History")
        plt.xlabel("Iteration")
        plt.ylabel("Distance")
        #show legend in the best location
        plt.legend(loc='best')
        plt.show()