import random
from abc import abstractmethod, ABC


class CrossoverMethod(ABC):

    def crossover(self,
                  parent_1: list,
                  parent_2: list
                  ) -> tuple[list]:
        assert len(parent_1) == len(parent_2), 'path lenghts must be the same'
        n = len(parent_1) - 1
        idx1, idx2 = sorted((random.randint(0, n), random.randint(0, n)))
        self._subset = slice(idx1, idx2)
        child_1, child_2 = self._get_children(parent_1=parent_1,
                                              parent_2=parent_2)
        first_child = self._modify(child_1=child_1,
                                   child_2=child_2)

        second_child = self._modify(child_1=child_2,
                                    child_2=child_1)
        return (first_child, second_child)

    @abstractmethod
    def _modify(self,
                child_1: list,
                child_2: list,
                ) -> list:
        pass

    def _get_children(self,
                      parent_1: list,
                      parent_2: list
                      ) -> tuple[list]:
        return parent_1, parent_2


class PMX(CrossoverMethod):
    def _get_children(self,
                      parent_1: list,
                      parent_2: list,
                      ) -> tuple[list]:
        sub = self._subset
        child_1, child_2 = parent_1, parent_2
        child_1[sub], child_2[sub] = child_2[sub], child_1[sub]
        return child_1, child_2

    def _modify(self,
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

    def _modify(self,
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
