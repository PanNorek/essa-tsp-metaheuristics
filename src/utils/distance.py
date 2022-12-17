import pandas as pd


def get_path_distance(path: list, distances: pd.DataFrame) -> int:
    """Calculate distance of the path based on distances matrix"""
    path_length = sum(distances.loc[x, y] for x, y in zip(path, path[1:]))
    # add distance back to the starting point
    path_length += distances.loc[path[0], path[-1]]
    return path_length
