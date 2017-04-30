import numpy as np

def read_data(path):
    data = np.genfromtxt("data_clean.txt", delimiter=" ")
    f2 = open('data_clean.txt', 'w')
    n = 14  # features number!!!!!
    #for arr in data:

    return data