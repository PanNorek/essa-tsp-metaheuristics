import pandas as pd
from ..utils import StopAlgorithm
from .switching_algorithm import SwitchingAlgorithm


class HillClimber(SwitchingAlgorithm):
    """Hill Climber Algorithm

    Parameters
    ----------
    neigh_type : str, optional
        Type of neighborhood, by default None
        Available options: 'swap', 'insert', 'inversion'
    n_iter : int, optional
        Number of iterations to perform, by default 30
        Algorithm will stop after n_iter iterations or when it reaches local minimum.
    verbose : bool, optional
        Print progress, by default False
    """

    NAME = "HILL CLIMBER"

    def _iterate_steps(self, distances: pd.DataFrame) -> None:
        """Iterate steps in Hill Climber Algorithm

        Args:
            distances (pd.DataFrame): Distance matrix
            swaps (List[tuple]): List of possible swaps

        Returns:
            distance (int): Total distance
            swap (tuple): Swap
        """
        self._next_iter()
        # get new path
        new_path = self._switch(distances=distances, how='best')
        # get new distance
        new_distance = self._get_path_distance(path=new_path, distances=distances)
        # distance gain
        gain = self.history[-1] - new_distance
        # break condition
        if gain <= 0:
            raise StopAlgorithm(iteration=self._i, distance=new_distance)
        if self._verbose:
            print(f"best switch: {self._last_switch_comment} - gain: {gain}")
            print(f"step {self._i}: distance: {new_distance}")

        # new path that shortens the distance
        self._path = new_path
        # adding new best distance to distances history
        self.history.append(new_distance)
