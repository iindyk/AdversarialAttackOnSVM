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
lll = []
lll.append(1)
print(lll)
n = 40  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0/n  # SVM regularization parameter
attack_size = 5  # random attack size
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
    if sum([p for p in point]) >= m*((B-A)/2):
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


def decompose_x(x):
    return np.array(x[:m]), x[m], \
           np.array(x[m+1:n+m+1]), np.array(x[n+m+1:2*n+m+1])   # w, b, l, a


def class_constr_eq(x, l_prev):
    ret = []
    w, b, l, a = decompose_x(x)
    for j in range(0, m):
        ret.append(w[j] - sum([l[i]*labels[i]*dataset[i][j] for i in range(0, n)]))
    ret.append(sum([l[i]*labels[i] for i in range(0, n)]))
    for i in range(0, n):
        ret.append(l[i] - l[i]*a[i] - l[i]*labels[i]*(np.dot(w, dataset[i]) + b))
        ret.append((C - l[i])*a[i])
    return ret


def class_constr_ineq(x):
    ret = []
    w, b, l, a = decompose_x(x)
    for i in range(0, n):
        ret.append(l[i])
        ret.append(C - l[i])
        ret.append(a[i])
        ret.append(labels[i]*(np.dot(w, dataset[i])+b)-1+a[i])
    return ret


def class_obj(x):
    return 1.0

options = {'maxiter': 100}
#x_opt = np.array([1.0/(2*n) for i in range(0, 2*n+m+1)])
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
# construct l:
l_opt = []
for i in range(0, n):
    if i in svc.support_:
        l_opt.append(svc.dual_coef_[0][list(svc.support_).index(i)])
    else:
        l_opt.append(0.0)
x_opt = list(svc.coef_[0]) + list(svc.intercept_) + l_opt + [0.0 for i in range(0, n)]
print(x_opt)
w_opt = x_opt[:m]
w_prev = [1.0 for j in range(0, m)]
l_prev = x_opt[m+1:n+m+1]
nit = 0
while (np.linalg.norm([w_prev[i] - w_opt[i] for i in range(0, m)]) > delta or not sol.success) and nit < maxit:
    print('iteration '+str(nit))
    w_prev = w_opt[:]
    con1 = {'type': 'eq', 'fun': class_constr_eq, 'args': [l_opt]}
    con2 = {'type': 'ineq', 'fun': class_constr_ineq}
    cons = [con1, con2]
    sol = minimize(class_obj, x_opt, constraints=cons, options=options, method='SLSQP')
    #print(class_constr_eq(sol.x))
    #print(min(class_constr_ineq(sol.x)))
    print(sol.success)
    print(sol.message)
    print(sol.nit)
    x_opt = list(sol.x)
    w_opt, b, l_prev, a = decompose_x(x_opt)
    nit += 1

print('w= ' + str(w_opt))
print('b= ' + str(b))
print('l= ' + str(l_prev))
print('a= ' + str(a))


svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labels_svc = svc.predict(dataset)
err_svc = 1 - accuracy_score(labels, predicted_labels_svc)
print('err on dataset by svc is '+str(err_svc))
predicted_labels_opt = np.sign([np.dot(dataset[i], w_opt)+b for i in range(0, n)])
err_opt = 1 - accuracy_score(labels, predicted_labels_opt)
print('err on dataset by opt is '+str(err_opt))