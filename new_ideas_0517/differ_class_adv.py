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
n = 30  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0/n  # SVM regularization parameter
attack_size = 0  # random attack size
A = 0
B = 100
eps = 0.1*(B-A)  # upper bound for norm of h
delta = 1e-2  # iteration precision level
maxit = 30  # iteration number limit
for i in range(0, n):
    point = []
    for j in range(0, m):
        point.append(uniform(A, B))
    dataset.append(point)
    # change
    if sum([p for p in point]) >= m*(B-A)/2:
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
predicted_labels = svc.predict(dataset)
err_orig = 1 - accuracy_score(labels, predicted_labels)
print('err on orig is '+str(err_orig))


def decompose_x(x):
    return np.array(x[:m]), x[m], np.array(x[m+1:m*n+m+1]), np.array(x[m*n+m+1:(m+1)*n+m+1]), \
           np.array(x[(m+1)*n+m+1:(m+2)*n+m+1]), np.array(x[(m+2)*n+m+1:(m+3)*n+m+1])   # w, b, h_hat, g, l, a

bnds = []
for j in range(0, (m+1)*n+m+1):
    bnds.append((-1000, 1000))
for i in range(0, n):
    bnds.append((0, C))
for i in range(0, n):
    bnds.append((0, 1000))


def adv_obj(x):
    av = 0.0
    for i in range(0, n):
        av += max(labels[i]*(np.dot(dataset[i], x[:m])+x[m]), -1.0)
    return av


def class_constr_inf_eq(x, w_prev, l_prev):
    ret = []
    w, b, h_hat, g, l, a = decompose_x(x)
    for j in range(0, m):
        ret.append(w[j] - sum([l[i]*labels[i]*dataset[i][j]+h_hat[j*n+i] for i in range(0, n)]))
    ret.append(sum([l[i]*labels[i] for i in range(0, n)]))
    for i in range(0, n):
        ret.append(l[i] - a[i] - l[i]*labels[i]*(np.dot(w, dataset[i]) + g[i] + b))
        hi = [h_hat[j*n +i] for j in range(0, m)]
        ret.append(1e-1*(np.dot(w_prev, hi) - l_prev[i]*g[i]))
        ret.append(l_prev[i]*a[i] - C*a[i])
    return ret


def class_constr_inf_eq_neg(x, w_prev, l_prev):
    ret = []
    w, b, h_hat, g, l, a = decompose_x(x)
    for j in range(0, m):
        ret.append(w[j] - sum([l[i]*labels[i]*dataset[i][j]+h_hat[j*n+i] for i in range(0, n)]))
    ret.append(sum([l[i]*labels[i] for i in range(0, n)]))
    for i in range(0, n):
        ret.append(l[i] - a[i] - l[i]*labels[i]*(np.dot(w, dataset[i]) + g[i] + b))
        hi = [h_hat[j*n +i] for j in range(0, m)]
        ret.append(1e-1*(np.dot(w_prev, hi) - l_prev[i]*g[i]))
        ret.append(l_prev[i]*a[i] - C*a[i])
    return -1.0*np.array(ret)


def class_constr_inf_ineq(x):
    ret = []
    w, b, h_hat, g, l, a = decompose_x(x)
    for i in range(0, n):
        ret.append(l[i])
        ret.append(C - l[i])
        ret.append(a[i])
        ret.append(labels[i]*(np.dot(w, dataset[i])+g[i]+b)-1+a[i]/C)
    ret.append(eps*(C**2)*n - np.dot(h_hat, h_hat))
    ret.append(1 - np.dot(w, w))
    return ret

l_opt = []
h_opt = []
for i in range(0, n):
    if i in svc.support_:
        #l_opt.append(1.0/n)
        l_opt.append(svc.dual_coef_[0][list(svc.support_).index(i)]/labels[i])
    else:
        l_opt.append(0.0)
x_opt = list(svc.coef_[0]) + list(svc.intercept_)+[0.0 for i in range(0, m*n+n)] + l_opt + [0.0 for i in range(0, n)]
w, b, h_hat, g, l, a = decompose_x(x_opt)
w_prev = np.array([0.5 for i in range(0, m)])
l_prev = np.array([0.5 for i in range(0, n)])
x_prev = np.array([0.5 for i in range(0, (m+3)*n+m+1)])
nit = 0
options = {'maxiter': 1000}
fl = True
while (adv_obj(x_prev) < adv_obj(x_opt) or fl
        #np.linalg.norm([w[i]-w_prev[i] for i in range(0, m)]) > delta
       #or np.linalg.norm([l[i]-l_prev[i] for i in range(0, n)]) > delta
       or not sol.success) and nit < maxit:
    fl = False
    x_prev = x_opt[:]
    w_prev = w[:]
    l_prev = l[:]
    print('iteration ' + str(nit))
    con1 = {'type': 'ineq', 'fun': class_constr_inf_eq, 'args': [w_prev, l_prev]}
    con1 = {'type': 'ineq', 'fun': class_constr_inf_eq_neg, 'args': [w_prev, l_prev]}
    con2 = {'type': 'ineq', 'fun': class_constr_inf_ineq}
    cons = [con1, con2]
    sol = minimize(adv_obj, x_opt, constraints=cons, options=options, method='COBYLA')
    print(sol.success)
    print(sol.message)
    x_opt = sol.x[:]
    w, b, h_hat, g, l, a = decompose_x(x_opt)
    print(w)
    print(b)
    #print(h_hat)
    #print(g)
    #print(l)
    #print(a)
    nit += 1

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

