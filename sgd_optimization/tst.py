import sgd_optimization.obj_con_functions_v2 as of2
from scipy.optimize import fsolve
import numpy as np
from sgd_optimization.random_dataset_generator import generate_random_dataset as grd

eps = 0.0
dataset, labels, colors = grd(read=True)
n = len(dataset)
m = len(dataset[0])
C = 1.0/n
h = np.zeros(n*m)


def classify(x):
    return of2.class_constr_all_eq(x[:m], x[m], h, x[m+1:m+n+1], x[m+n+1:m+2*n+1], dataset, labels, eps, C)

x0 = np.zeros(m+2*n+1)
sol = fsolve(classify, x0)
print(sol.x)
print(sol.infodict)
print(sol.mesg)