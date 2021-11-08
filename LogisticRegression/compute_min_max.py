import numpy as np

def compute_min_max(data_path, first_col_index=1, last_col_index=23):
    data = np.loadtxt(data_path, delimiter=",", dtype=np.float32, skiprows=1)[:, first_col_index:last_col_index]
    return np.stack((data.min(axis=0), data.max(axis=0)))
