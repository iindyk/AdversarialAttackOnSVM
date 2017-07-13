import sgd_optimization.obj_con_functions_v2 as of2
from scipy.optimize import root
import numpy as np
from sgd_optimization.random_dataset_generator import generate_random_dataset as grd
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sgd_optimization.optimize as opt

A = np.zeros((2, 2))
A[0, 0] = 1
A[0, 1] = 1
print(A)

