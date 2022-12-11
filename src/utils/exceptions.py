
class StopAlgorithm(Exception):
    """Exception raised when algorithm stops."""
    def __init__(self, iteration: int, distance: int, *args: object) -> None:
        super().__init__(*args)
        self.message = f'Algorithm stopped at {iteration} iteration\nBest distance: {distance}'
