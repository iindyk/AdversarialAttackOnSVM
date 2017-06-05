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
n = 50  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0/n  # SVM regularization parameter
attack_size = 10  # random attack size
A = 0
B = 100
eps = 0.1*(B-A)  # upper bound for norm of h
delta = 1e-7  # iteration precision level
maxit = 100  # iteration number limit
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
for i in range(0, attack_size):
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
print('err on orig is '+str(err_orig))


def decompose_x(x):
    return np.array(x[:m]), x[m], np.array(x[m+1:m*n+1]), np.array(x[m*n+1:(m+1)*n+1]), \
           np.array(x[(m+1)*n+1:(m+2)*n+1]),np.array(x[(m+2)*n+1:(m+3)*n+1])   # w, b, h_hat, g, l, a


def adv_obj(x):
    av = 0.0
    for i in range(0, n):
        av += max(labels[i]*(np.dot(dataset[i], x[:m])+x[m]), -1.0)
    return av/n


def class_constr_inf_eq(x, h_prev, l_prev):
    ret = []
    w, b, h_hat, g, l, a = decompose_x(x)
    for j in range(0, m):
        ret.append(w[j] - C*sum([l[i]*labels[i]*dataset[i][j]+h_hat[j*n+i] for i in range(0, n)]))
    ret.append(sum([l[i]*labels[i] for i in range(0, n)]))
    for i in range(0, n):
        ret.append(l[i] - a[i] - l_prev[i]*labels[i]*(np.dot(w, dataset[i]) + g[i] + b))
        hi_prev = [h_prev[j*n +1] for j in range(0, m)]
        ret.append(np.dot(w, hi_prev) - l_prev[i]*g[i])
    return ret


def class_constr_inf_ineq(x):
    ret = []
    av = 0.0
    w, b, h_hat, g, l, a = decompose_x(x)
    for i in range(0, n):
        ret.append(l[i])
        ret.append(1.0/n - l[i])
        ret.append(labels[i]*(np.dot(w, dataset[i])+g[i]+b)-1+n*a[i])
    ret.append(eps/n - np.dot(h_hat, h_hat))
    return ret


x_opt = np.array([1.0 for i in range(0, (m+3)*n+1)])
w, b, h_hat, g, l, a = decompose_x(x_opt)
h_prev = np.array([0.0 for i in range(0, m*n)])
l_prev = np.array([0.0 for i in range(0, n)])
nit = 0
while (np.linalg.norm(h_hat - h_prev) > delta or np.linalg.norm(l-l_prev) > delta) and nit<maxit:
    h_prev = h_hat[:]
    l_prev = l[:]
    print('iteration ' + str(nit))
    con1 = {'type': 'eq', 'fun': class_constr_inf_eq, 'args': [h_prev, l_prev]}
    con2 = {'type': 'ineq', 'fun': class_constr_inf_ineq}
    cons = [con1, con2]
    sol = minimize(adv_obj, x_opt, constraints=cons)
    print(sol.success)
    print(sol.message)
    x_opt = sol.x
    w, b, h_hat, g, l, a = decompose_x(x_opt)

dataset_infected = []
h = []
for j in range(0, m):
    for i in range(0, n):
        h.append(0 if l[i] == 0 else h_hat[i+j*n]/l[i])
for i in range(0, n):
    temp = []
    for j in range(0, m):
        temp.append(dataset[i][j] + h[j * n + i])
        dataset_infected.append(temp)
svc = svm.SVC(kernel='linear', C=C).fit(dataset_infected, labels)
predicted_labels_inf_svc = svc.predict(dataset)
err_inf_svc = 1 - accuracy_score(labels, predicted_labels_inf_svc)
print('err on infected dataset by svc is '+str(err_inf_svc))
predicted_labels_inf_opt = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
err_inf_opt = 1 - accuracy_score(labels, predicted_labels_inf_opt)
print('err on infected dataset by opt is '+str(err_inf_opt))

