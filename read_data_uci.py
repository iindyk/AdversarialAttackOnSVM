import numpy as np


def read_data_uci(path):
    data_arr = np.genfromtxt(path, delimiter=",", dtype=float)
    data = data_arr.tolist()
    for ar in data:
        if ar.pop() == 1.0:
            ar.insert(0, 1.0)
        else:
            ar.insert(0, -1.0)

    return data
