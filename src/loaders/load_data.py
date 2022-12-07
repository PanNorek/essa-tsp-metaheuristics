import pandas as pd
import numpy as np

def load_data(path: str, triu=False):
    # Load data from path with first column as index
    data = pd.read_excel(path, index_col=0, header=[0])

    if triu:
        # Get upper triangle of data
        return pd.np.triu(data.to_numpy())
 
    return data
    # return data.to_numpy()
