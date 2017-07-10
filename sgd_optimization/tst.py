import sgd_optimization.obj_con_functions_v2 as of2
from scipy.optimize import root
import numpy as np
from sgd_optimization.random_dataset_generator import generate_random_dataset as grd
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sgd_optimization.optimize as opt

n = 200
A = np.zeros((n-5, n))
for i in range(0, n-5):
    A[i, i] = 1.0
    A[n-6-i, i] = -1.0
for i in range(0, 40):
    np.append(A, A[0])
b = np.zeros(n-5)
y = np.ones(n)
y_proj = opt.project_subspace(y, A, b)
print(y_proj)
print(np.max(np.dot(A, y_proj)))
