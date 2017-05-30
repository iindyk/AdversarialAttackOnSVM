import numpy as np
from scipy.optimize import minimize

from random import uniform
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
n = 50  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0/n  # SVM regularization parameter
a = 10  # random attack size
A = 0
B = 100
eps = 0.1*(B-A)  # upper bound for norm of h
for i in range(0, n):
    point = []
    for j in range(0, m):
        point.append(uniform(A, B))
    dataset.append(point)
    # change
    if sum([p**2 for p in point]) >= 5000:
        labels.append(1.0)
        colors.append((1, 0, 0))
    else:
        labels.append(-1.0)
        colors.append((0, 0, 1))

# random attack
for i in range(0, a):
    if labels[i] == 1:
        labels[i] = -1
        colors[i] = (0, 0, 1)
    else:
        labels[i] = 1
        colors[i] = (1, 0, 0)
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
x_svc = list(svc.coef_[0])
x_svc.append(svc.intercept_[0])
predicted_labels = svc.predict(dataset)
err_orig = 1 - accuracy_score(labels, predicted_labels)


def class_obj_inf(x, g):
    av = 0.0
    for i in range(0, n):
        av += max(1-labels[i]*(np.dot(dataset[i], x[:m])+x[m]+g[i]), 0)
    return np.dot(x[:m], x[:m])/2.0 + C*av
x0 = np.array([1 for i in range(0, m+1)])
w = x0[:m]
b = x0[m]


def clas_obj_min(g):
    sol = minimize(class_obj_inf, x0, args=[g])
    w = sol.x[:m]
    b = sol.x[m]
    if not sol.success:
        print(sol.message)
        print('nit = ' + str(sol.nit))
    return -sol.fun


def adv_norm_constr(g):
    return -sum([g[i]**2])/n + np.dot(w, w)*(eps**2)


con1 = {'type': 'ineq', 'fun': adv_norm_constr}
cons = [con1]
g0 = np.array([1 for i in range(0, n)])
sol_adv = minimize(clas_obj_min, g0, constraints=cons)
print(sol_adv.message)
print(sol_adv.nit)
g = sol_adv.x


def obj_h(x):
    return np.dot(x, x)


def constr_h(x, w, g):
    ret = []
    for i in range(0, n):
        ret.append(x[i]*w[0]+x[n+i]*w[1]-g[i])
    return ret

h0 = np.array([1 for i in range(0, 2 * n)])
con_h = {'type': 'eq', 'fun': constr_h, 'args': [w_opt, g_opt]}
cons_h = ([con_h])
sol_h = minimize(obj_h, h0, bounds=None, constraints=cons_h)
h = list(sol_h.x)
dataset_infected = []
for i in range(0, n):
    temp = []
    for j in range(0, m):
        temp.append(dataset[i][j] + h[j * n + i])
    dataset_infected.append(temp)
svc_inf = svm.SVC(kernel='linear', C=C).fit(dataset_infected, labels)
pr_lb = svc_inf.predict(dataset)
err_inf = 1 - accuracy_score(labels, pr_lb)
print('err_orig = ' + str(err_orig))
print('err_inf = ' + str(err_inf))