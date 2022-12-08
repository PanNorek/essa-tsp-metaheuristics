import pandas as pd
import numpy as np
from typing import Union


def load_data(path: str,
              triu: bool = False,
              as_array: bool = False
              ) -> Union[pd.DataFrame, np.array]:
    # Load data from path with first column as index
    data = pd.read_excel(path, index_col=0, header=[0])
    if triu:
        # Get upper triangle of data
        return pd.np.triu(data.to_numpy())
    if as_array:
        return data.to_numpy()
    return data
