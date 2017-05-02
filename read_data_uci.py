import numpy as np


def read_data_uci_ionosphere(path):
    data_arr = np.genfromtxt(path, delimiter=",", dtype=str)
    data_str = data_arr.tolist()
    data_int = []
    for i in range(0, len(data_str)):
        data_int.append([])
        if data_str[i].pop() == 'g':
            data_int[i].append(1.0)
        else:
            data_int[i].append(-1.0)
        for p in data_str[i]:
            data_int[i].append(float(p))
    return data_int


def read_data_uci_breast_cancer(path):
    data_arr = np.genfromtxt(path, delimiter=",", dtype=float)
    data = data_arr.tolist()
    for ar in data:
        ar[0] -= 1000000
        ar[0] /= 1000
        if ar.pop() == 2.0:
            ar.insert(0, 1.0)
        else:
            ar.insert(0, -1.0)
    return data


def read_data_uci_survival(path):
    data_arr = np.genfromtxt(path, delimiter=",", dtype=float)
    data = data_arr.tolist()
    for ar in data:
        if ar.pop() == 1.0:
            ar.insert(0, 1.0)
        else:
            ar.insert(0, -1.0)
    return data


def read_data_uci_spectf(path):
    data_arr = np.genfromtxt(path, delimiter=",", dtype=float)
    data = data_arr.tolist()
    for ar in data:
        if ar[0] == 0:
            ar[0] = -1.0
    return data
