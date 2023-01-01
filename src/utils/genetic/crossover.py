import random
from abc import abstractmethod, ABC
from ...utils import path_check


class CrossoverMethod(ABC):
    """
    Interface for Crossover methods used in Genetic Algorithm

    Methods:
        crossover - crosses over two parents represented by two list
            that correspond to paths in TSP algorithms.
            Returns two resulting paths representing children
            that will be a part of new generation

    Crossover, also called recombination, is a genetic operator used to
    combine the genetic information of two parents to generate new offspring.
    It is one way to stochastically generate new solutions from an existing population

    Paths in crossover method are lists that represent the chromosomes
    of two chosen parents.

    Chromosome is a set of parameters which define a proposed solution
    to the problem that the genetic algorithm is trying to solve

    In TSP chromosome is represented by the path (order of the cities)
    One of the hindrances is a requirement of uniqueness. Genes must not
    repeat in the chromosome. That is why only specific crossover methods
    apply in TSP. One of them are PMX and OX that are implemented.
    Lists are combined in a method specific way and produce two
    new paths that correspond to children chromosomes.

    Check out:

    src.algos.genetic_algorithm GeneticAlgorithm

    For more information check out:
    https://en.wikipedia.org/wiki/Genetic_algorithm
    https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)
    https://en.wikipedia.org/wiki/Chromosome_(genetic_algorithm)
    """
    NAME = None

    def crossover(self, parent_1: list, parent_2: list) -> tuple[list]:
        """
        Performs method specific a crossover of two parent chromosomes
        and returns child chromosomes

        Params:
            parent_1: list
                Chromosome of the first parent
            parent_2: list
                Chromosome of the second parent

        Parent chromosomes are the lists representing paths,
        the order in which salesman visits the cities.
        """
        # chromosomes must be of the same length to perform crossover
        assert len(parent_1) == len(parent_2), "path lengths must be the same"
        n = len(parent_1) - 1
        # two crossover points are picked randomly from the parent chromosomes
        idx1, idx2 = sorted((random.randint(0, n), random.randint(0, n)))
        # slice object is created from crossover points
        self._subset = slice(idx1, idx2)
        # get child chromosomes before their modification
        # ex. in PMX subsets between crossover points are exchanged between parent chromosomes
        child_1, child_2 = self._get_children(parent_1=parent_1, parent_2=parent_2)
        # modifying first child - method specific modifications and legalizations
        first_child = self._modify(child_1=child_1, child_2=child_2)
        # check uniqueness invariant
        assert (
            len(set(first_child)) == len(first_child)
        ), "Crossover failed - elements in child chromosome are not unique"
        # modifying second child
        second_child = self._modify(child_1=child_2, child_2=child_1)
        # check uniqueness invariant
        assert (
            len(set(second_child)) == len(second_child)
        ), "Crossover failed - elements in child chromosome are not unique"
        # returns a tuple of child chromosomes
        return (first_child, second_child)

    @abstractmethod
    def _modify(
        self,
        child_1: list,
        child_2: list,
    ) -> list:
        """
        Performs method specific modification of two child chromosomes
        and their legalization for TSP path uniqueness requisite.

        Params:
            child_1: list
                Chromosome of the first child
            child_2: list
                Chromosome of the second child

        Returns:
            Legal child chromosomes

        Child chromosomes are the lists representing paths,
        the order in which salesman visits the cities.
        """
        pass

    def _get_children(self, parent_1: list, parent_2: list) -> tuple[list]:
        """
        Returns premodified child chromosomes from parent chromosomes
        specific to given crossover method.

        parent_1: list
            Chromosome of the first parent
        parent_2: list
            Chromosome of the second parent

        Example:
         In PMX subsets between crossover points are exchanged
         between parent chromosomes.

        In most cases returns a copy of parent chromosomes

        Parent chromosomes are the lists representing paths,
        the order in which salesman visits the cities.
        """
        return parent_1[:], parent_2[:]


class PMX(CrossoverMethod):
    """
    Partially Mapped Crossover (PMX)

     A way to combine two individuals chromosome resulting in two new child chromosome.
        - carve out randomly selected slice of each parent and replace it in its counterpart
        - ensure that the "unique item" invariant for both child individuals is maintained

    Methods:
        crossover - crosses over two parents represented by two list
            that correspond to paths in TSP algorithms.
            Returns two resulting paths representing children
            that will be a part of new generation

    Implements:
        CrossoverMethod - general interface for crossover methods

    Check out:

    src.algos.genetic_algorithm GeneticAlgorithm
    """
    NAME = "PMX"

    def _get_children(
        self,
        parent_1: list,
        parent_2: list,
    ) -> tuple[list]:
        # in PMX subsets between crossover points are exchanged between parent chromosomes
        sub = self._subset
        child_1, child_2 = parent_1[:], parent_2[:]
        child_1[sub], child_2[sub] = child_2[sub], child_1[sub]
        return child_1, child_2

    def _modify(
        self,
        child_1: list,
        child_2: list,
    ) -> list:
        child_1, child_2 = child_1[:], child_2[:]
        for i, gene in enumerate(child_1):
            # if gene is included in chosen subset - skip
            # child = [3, 4, 8, |1, 6, 8|, 6, 5]
            # child2= [4, 2, 5, |2, 7, 1|, 3, 7]
            # |1, 6, 8| stays without any change
            # 8 -> 1 -> 2 since 1 is still duplicated
            # 6 -> 7
            if i in range(*self._subset.indices(len(child_1))):
                continue
            # LEGALIZING - making unique path
            # until gene from outside of the self._subset is the same as one within self._subset
            # change it with ist mappping from other child
            while gene in child_1[self._subset]:
                idx = child_1[self._subset].index(gene)
                gene = child_2[self._subset][idx]
            child_1[i] = gene
        return child_1


class OX(CrossoverMethod):
    """
    Order Crossover (OX)

    A variation of PMX with a different repairing procedure

    Methods:
        crossover - crosses over two parents represented by two list
            that correspond to paths in TSP algorithms.
            Returns two resulting paths representing children
            that will be a part of new generation

    Implements:
        CrossoverMethod - general interface for crossover methods

    Check out:

    src.algos.genetic_algorithm GeneticAlgorithm

    src.utils.genetic.crossover PMX
    """
    NAME = "OX"

    def _modify(
        self,
        child_1: list,
        child_2: list,
    ) -> list:
        child_1, child_2 = child_1[:], child_2[:]
        # clear entire list except for chosen subset
        child_1 = [x if x in child_1[self._subset] else 0 for x in child_1]

        idx2 = self._subset.indices(len(child_1))[1]
        child_2_iter = child_2[idx2:] + child_2[:idx2]
        to_fill_idx = idx2

        for gene in child_2_iter:
            # child = [3, 4, 8, |2, 7, 1|, 6, 5]
            # child2= [4, 2, 5, |1, 6, 8|, 3, 7]
            # |2, 7, 1| stays without any change
            # rest of the first list is filled starting from right side of the subset
            # with genes from the second list starting from right side of the subset
            # 3, 7, 4, 2, 5, 1, 6, 8 -> if they are not in first list
            if gene in child_1:
                continue
            child_1[to_fill_idx] = gene
            if (to_fill_idx + 1) == len(child_1):
                to_fill_idx = 0
            else:
                to_fill_idx += 1
        return child_1
