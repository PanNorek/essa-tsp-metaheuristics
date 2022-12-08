import pandas as pd
import numpy as np
from typing import Union

def load_data(path: str, to_numpy: bool = False) -> Union[pd.DataFrame, np.ndarray]:
    """Load data from given path

    Args:
        path (str): Path to data

    Returns:
        data : pd.DataFrame or np.ndarray
    """

    # Load data from path with first column as index
    data = pd.read_excel(path, index_col=0)

    
    if to_numpy:
        return data.to_numpy()
    
    return data
    
