import numpy as np
from scipy.optimize import minimize, root

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
delta = 1e-3  # iteration precision level
maxit = 100  # iteration number limit
for i in range(0, n):
    point = []
    for j in range(0, m):
        point.append(uniform(A, B))
    dataset.append(point)
    # change
    if sum([p**2 for p in point]) >= m*n*(B-A)/2:
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
    return np.array(x[:m]), x[m], \
           np.array(x[m+1:n+m+1]), np.array(x[n+m+1:2*n+m+1])   # w, b, l, a


def class_constr_eq(x):
    ret = []
    w, b, l, a = decompose_x(x)
    for j in range(0, m):
        ret.append(w[j] - C*sum([l[i]*labels[i]*dataset[i][j] for i in range(0, n)]))
    ret.append(sum([l[i]*labels[i] for i in range(0, n)]))
    for i in range(0, n):
        ret.append(l[i] - a[i] - l[i]*labels[i]*(np.dot(w, dataset[i]) + b))
    return ret


def class_constr_ineq(x):
    ret = []
    w, b, l, a = decompose_x(x)
    for i in range(0, n):
        ret.append(l[i])
        ret.append(1.0/n - l[i])
        ret.append(a[i])
        ret.append(labels[i]*(np.dot(w, dataset[i])+b)-1+n*a[i])
    return ret


def class_obj(x):
    return sum(x[n+m+1:2*n+m+1])

x0 = np.array([1.0/n for i in range(0, 2*n+m+1)])
con1 = {'type': 'eq', 'fun': class_constr_eq}
con2 = {'type': 'ineq', 'fun': class_constr_ineq}
cons = [con1, con2]
sol = minimize(class_obj, x0, constraints=cons)
print(sol.success)
print(sol.message)
w, b, l, a = decompose_x(sol.x)
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labels_svc = svc.predict(dataset)
err_svc = 1 - accuracy_score(labels, predicted_labels_svc)
print('err on dataset by svc is '+str(err_svc))
predicted_labels_opt = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
err_opt = 1 - accuracy_score(labels, predicted_labels_opt)
print('err on dataset by opt is '+str(err_opt))