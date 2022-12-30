# essa-tsp-metaheuristics

📊 About

A repository containing heuristics for the traveling salesman problem (TSP) is a collection of algorithms and techniques that can be used to approximately solve the TSP. The TSP is a classic optimization problem in computer science that involves finding the shortest possible route that visits a given set of cities and returns to the starting city. The problem is NP-hard, meaning that it is very difficult to find an exact solution, particularly for large sets of cities. Heuristics are methods that can quickly find a good, but not necessarily optimal, solution to the TSP. These methods are useful for real-world applications where finding the exact solution may be impractical due to time or resource constraints. The repository contains a variety of heuristics, such as nearest neighbor, hill climber and genetic algorithm.

## Contents
- [x] Nearest Neighbour Algorithm
- [x] Hill Climber Algorithm
- [x] Tabu Search Algorithm
- [x] Simulated Annealing Algorithm
- [x] Genetic Algorithm

## 📋 Requirements
  1. The library works under Python 3.9+
  2. Algorithms need distance matrix to solve the problem. 

## ⭐ Features
- [x] Parallel Multistart for algorithms except Genetic Algorithm
- [x] GridSearch running in parallel for Genetic Algorithm

⭐ **Parallel Multistart**

```python
from src.utils import load_data, PathPlotter, BenchmarkPlotter, DistanceHistoryPlotter
from src.algos import *
import os

distances = load_data(os.path.join("data", "Data_TSP_127.xlsx"))

result = MultistartAlgorithm()(HillClimbing, distances, n_starts=NUM_STARTS, only_best=True, verbose=False, n_jobs=-1, n_iter=25)
print(result)
```

⭐ **GridSearch**

```python
from src.utils import *
from src.algos import *
import os

distances = load_data(os.path.join("data", "Data_TSP_29.xlsx"))

param_grid = {
    "POP_SIZE": [500],
    "N_ITERS": [100],
    "SELECTION_METHOD": ["truncation", "roulette", "tournament"], # "truncation", "roulette", "tournament"
    "CROSSOVER_METHOD": ["pmx", "ox"], # "ox", "pmx" 
    "ELITE_SIZE": [0, ],
    "MATING_POOL_SIZE": [.5],
    "MUTATION_RATE": [.15],
    "NEIGH_TYPE": ["inversion", "insertion", "swap"], # "inversion", "insertion", "swap"
    "VERBOSE": [True]
}

gg = GridSearchGA(param_grid=param_grid, n_jobs=-1)
result = gg.solve(distances)
print(result)
```

## 📝 Examples
**Example 1. Run Nearest Neighbor Algorithm**

The nearest neighbor algorithm is a heuristic for solving the traveling salesman problem (TSP). It is a simple and efficient method that can be used to quickly find a good, although not necessarily optimal, solution to the TSP. The algorithm works by iteratively adding the nearest unvisited city to the current tour until all cities have been visited.
```python

from src.utils import load_data, PathPlotter, BenchmarkPlotter, DistanceHistoryPlotter
from src.algos import *
import os

distances = load_data(os.path.join("data", "Data_TSP_127.xlsx"))

n = NearestNeighbour(verbose=False)
result = n.solve(distances, start_order=14) # we start from 14th city
print(result)
```

## 👨‍💻 Contributing
* [sewcio543](https://github.com/sewcio543)


## 📂 Directory Structure
    ├───.github
    |   └───workflows
    ├───data
    ├───notebooks
    └───src
        ├───algos
        └───utils
            └───genetic

## 📅 Development schedule
**Version 1.0.0**

- [x] Nearest Neighbour Algorithm
- [x] Hill Climber Algorithm
- [x] Tabu Search Algorithm
- [x] Simulated Annealing Algorithm
- [x] Genetic Algorithm
- [x] Parallel Multistart for algorithms except Genetic Algorithm
- [x] GridSearch running in parallel for Genetic Algorithm
- [ ] Python package

**Version 2.0.0**

- [ ] Rework Multistart feature
- [ ] Code optimization
- [ ] More data load possibilities


## 🎓 Learning Materials
* [The Travelling Salesman Problem](https://classes.engr.oregonstate.edu/mime/fall2017/rob537/hw_samples/hw2_sample2.pdf)
* [Hill Climber Algorithm](https://classes.engr.oregonstate.edu/mime/fall2017/rob537/hw_samples/hw2_sample2.pdf)
* [About Genetic Oerations](https://mat.uab.cat/~alseda/MasterOpt/GeneticOperations.pdf)


## 📧 Contact
[![](https://img.shields.io/twitter/url?label=/rafal-nojek/&logo=linkedin&logoColor=%230077B5&style=social&url=https%3A%2F%2Fwww.linkedin.com%2in%2rafaln97%2F)](https://www.linkedin.com/in/rafaln97/) [![](https://img.shields.io/twitter/url?label=/PanNorek&logo=github&logoColor=%23292929&style=social&url=https%3A%2F%2Fgithub.com%2FPanNorek)](https://github.com/PanNorek)

[![](https://img.shields.io/twitter/url?label=/sewcio543&logo=github&logoColor=%23292929&style=social&url=https%3A%2F%2Fgithub.com%2Fsewcio543)](https://github.com/sewcio543)

## 📄 License

GNU GENERAL PUBLIC LICENSE

