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
C = 1.0  # SVM regularization parameter
a = 0  # random attack size
A = 0
B = 100
eps = 0.1*(B-A)  # upper bound for norm of h
delta = 1e-4  # iteration precision level
maxit = 50  # iteration number limit
for i in range(0, n):
    point = []
    for j in range(0, m):
        point.append(uniform(A, B))
    dataset.append(point)
    # change
    if sum([p for p in point]) >= 100:
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


def adv_obj(x):
    av = 0.0
    for i in range(0, n):
        av += max(labels[i]*(np.dot(dataset[i], x[:m])+x[m]), -1.0)
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


def class_constr(x, p, q):
    ret = []
    ret.append(q*(class_obj_orig(x_svc)+C*n*(np.dot(x_svc[:m], x_svc[:m]))**0.5 * eps) - class_obj_inf(x))
    ret.append(class_obj_inf(x) - p*(class_obj_orig(x_svc)+C*n*(np.dot(x_svc[:m], x_svc[:m]))**0.5 * eps))
    return ret


def obj_h(x):
    return np.dot(x, x)


def constr_h(x, w, g):
    ret = []
    for i in range(0, n):
        ret.append(x[i]*w[0]+x[n+i]*w[1]-g[i])
    return ret

# iterative scheme
nit = 0
w_svc = np.array([0.0 for i in range(0, m)])
w_opt = np.array([1.0 for i in range(0, m)])
x_opt0 = np.array([1.0 for i in range(0, m+n+1)])
u = 0.0
v = 1.0
options = {'maxiter': 100}
while (w_svc[0]-w_opt[0])**2 + (w_svc[1]-w_opt[1])**2 > delta and nit < maxit:
    print('iteration '+str(nit))
    con1 = {'type': 'ineq', 'fun': attack_norm_constr}
    con2 = {'type': 'ineq', 'fun': class_constr, 'args': [u, v]}
    cons = [con1, con2]
    sol = minimize(adv_obj, x_opt0, bounds=None, options=options, constraints=cons)
    x_opt = list(sol.x)
    w_opt = sol.x[:m]
    b_opt = sol.x[m]
    g_opt = sol.x[m+1:]
    print(sol.message)
    if not sol.success:
        print('u = '+str(u)+' v = '+str(v)+' no sol')
        tmp = u
        u = v
        v = 2*v - tmp
    else:
        print('u = ' + str(u) + ' v = ' + str(v) + ' exist sol')
        v = (u+v)/2
    sol.clear()
    # restoring h
    h0 = np.array([0.1 for i in range(0, 2 * n)])
    con_h = {'type': 'eq', 'fun': constr_h, 'args': [w_opt, g_opt]}
    cons_h = ([con_h])
    sol_h = minimize(obj_h, h0, bounds=None, constraints=cons_h)
    h = list(sol_h.x)
    sol_h.clear()
    dataset_infected = []
    for i in range(0, n):
        temp = []
        for j in range(0, m):
            temp.append(dataset[i][j] + h[j * n + i])
        dataset_infected.append(temp)
    svc = svm.SVC(kernel='linear', C=C).fit(dataset_infected, labels)
    w_svc = svc.coef_[0]
    b_svc = svc.intercept_[0]
    nit += 1

predicted_labels_inf = np.sign([np.dot(dataset[i], w_opt)+b_opt for i in range(0, n)])
err_inf = 1 - accuracy_score(labels, predicted_labels_inf)
print(nit)
print(w_opt)
print(b_opt)
print(w_svc)
print(b_svc)
print(err_orig)
print(err_inf)
plt.subplot(221)
plt.title('original')
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.subplot(222)
plt.title('infected')
plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
plt.show()