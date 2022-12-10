
class StopAlgorithm(Exception):
    def __init__(self, iter: int, *args: object) -> None:
        super().__init__(*args)
        self.message = f'Algorithm stopped at {iter} iteration'
