from typing import Union
import pandas as pd
import numpy as np


def load_data(path: str,
              triu: bool = False,
              as_array: bool = False
              ) -> Union[pd.DataFrame, np.array]:
    """
    Load data from given path

    Args:
        path (str): Path to data

    Returns:
        data : pd.DataFrame or np.ndarray
    """
    data = pd.read_excel(path, index_col=0)
    if triu:
        # Get upper triangle of data
        return pd.np.triu(data.to_numpy())
    return data.to_numpy() if as_array else data
