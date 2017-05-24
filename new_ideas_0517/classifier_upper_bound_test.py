import numpy as np
from scipy.optimize import minimize

from random import uniform
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
colors_h = []
n = 200  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0  # SVM regularization parameter
a = 2  # random attack size
A = 10
B = 110
eps = 0.1*(B-A)  # upper bound for norm of h
delta = 0.1  # iteration precision level
maxit = 20
for i in range(0, n):
    point = []
    for j in range(0, m):
        point.append(uniform(A, B))
    dataset.append(point)
    # change
    if sum([p**2 for p in point]) >= 75**2:
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
svm = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
x_svc = list(svm.coef_[0])
x_svc.append(svm.intercept_[0])


def adv_obj(x):
    av = 0.0
    for i in range(0, n):
        av += max(1+labels[i]*(np.dot(dataset[i], x[:m])+x[m]), 0)
    return av/n


def class_obj_inf(x):
    av = 0.0
    for i in range(0, n):
        av += max(1-labels[i]*(np.dot(dataset[i], x[:m])+x[m]+x[m+1+i]), 0)
    return np.dot(x[:m], x[:m])/2.0 + C*av


def class_obj_orig(x):
    av = 0.0
    for i in range(0, n):
        av += max(1-labels[i]*(np.dot(dataset[i], x[:m])+x[m]), 0)
    return np.dot(x[:m], x[:m])/2.0 + C*av


def attack_norm_constr(x):
    ret = []
    norm = np.dot(x[:m], x[:m])*(eps**2)
    for i in range(0, n):
        ret.append(-x[m+1+i]**2 + norm)
    return ret


def class_constr(x, u, v):
    ret = []
    ret.append(v*(class_obj_orig(x_svc)+C*n*np.dot(x_svc[:m], x_svc[:m])**0.5*eps) - class_obj_inf(x))
    ret.append(class_obj_inf(x) - u*(class_obj_orig(x_svc)+C*n*np.dot(x_svc[:m], x_svc[:m])**0.5*eps))


def obj_h(h):
    return np.dot(h, h)


def constr_h(h, w, g):
    ret = []
    for i in range(0, n):
        ret.append(h[i]*w[0]+h[n+i]*w[1]-g[i])
    return ret

# iterative scheme
nit = 0
w_svc = np.array([0.0 for i in range(0, m)])
w_opt = np.array([1.0 for i in range(0, m)])
u = 0.0
v = 1.0
while (w_svc[0]-w_opt[0])**2 + (w_svc[1]-w_opt[1])**2 > delta and nit < maxit:
    con1 = {'type': 'ineq', 'fun': attack_norm_constr}
    con2 = {'type': 'ineq', 'fun': class_constr, 'args': [u, v]}
    cons = [con1, con2]
    sol = minimize(adv_obj, w_opt, bounds=None, constraints=cons)
    w_opt = sol.x[:m]
    b_opt = sol.x[m]
    g_opt = sol.x[m+1:]
    if not sol.success:
        tmp = u
        u = v
        v = 2*v - u
    else:
        v = (u+v)/2
    sol.clear()
    # restoring h
    h0 = np.array([0.1 for i in range(0, 2 * n)])
    con_h = {'type': 'eq', 'fun': constr_h, 'args': [w_opt, g_opt]}
    cons_h = ([con_h])
    sol_h = minimize(obj_h, h0, bounds=None, constraints=cons_h)
    h = sol_h.x
    sol_h.clear()
    dataset_infected = []
    for i in range(0, n):
        temp = []
        for j in range(0, m):
            temp.append(dataset[i][j] + h[j * n + i])
        dataset_infected.append(temp)
    svm = svm.SVC(kernel='linear', C=C).fit(dataset_infected, labels)
    w_svc = svm.coef_[0]
    b_svc = svm.intercept_[0]
    nit += 1

print(nit)
print(w_opt)
print(b_opt)
print(w_svc)
print(b_svc)