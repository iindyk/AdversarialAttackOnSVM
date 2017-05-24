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
n = 400  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0  # SVM regularization parameter
a = 2  # random attack size
A = 10
B = 110
eps_list = [(i/100.0)*(B-A) for i in range(20, 25)]  # upper bound for norm of h
h_list = []  # attacks, first is zero attack
nsim = 50  # number of simulations
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

# generating random attacks
temp = np.random.normal(0.0, 1.0, nsim*n*2)
for i in range(0, n*nsim):
    t = np.array([])
    nrm = 0.0
    for j in range(0, m):
        t = np.append(t, temp[i + j * nsim * n])
        nrm += temp[i + j * nsim * n]**2
    h_list.append([t[k]/nrm**0.5 for k in range(0, m)])
# separation on nsim lists
h_d_list = []
for i in range(0, nsim):
    h_d_list.append([h_list[i + j * nsim] for j in range(0, n)])
# c-svm optimization
errs = []
maxerrs = []
maxerr_hs = []
for e in eps_list:
    maxerr = 0.0
    maxerr_h = []
    for h in h_d_list:
        infected_dataset = [[dataset[j][i] + e * h[j][i] for i in range(0, m)] for j in range(0, n)]
        svc = svm.SVC(kernel='linear', C=C).fit(infected_dataset, labels)
        predicted_labels = svc.predict(dataset)
        err = 1 - accuracy_score(labels, predicted_labels)
        errs.append(err)
        if err > maxerr:
            maxerr = err
            maxerr_h = h
    maxerrs.append(maxerr)
    maxerr_hs.append(maxerr_h)
    colors_h.append((0, 1, 0))
print('svc is done')

# subdifferential adversarial attack
def class_constr(x):
    res = []
    for i in range(0, m):
        summ = 0.0
        for j in range(0, n):
            # summ += labels[j] * dataset[j][i] * (1 if labels[j]*(np.dot(dataset[j], x[:m])+x[m]) < 1 else 0)
            # summ += labels[j]*dataset[j][i]*max(1-labels[j]*(np.dot(dataset[j], x[:m])+x[m]),0)
            summ += labels[j] * dataset[j][i] * (1 - labels[j] * (np.dot(dataset[j], x[:m]) + x[m]+x[m+1+j]))
        res.append(x[i] - C*(1.0/n)*summ)
    av = 0.0
    for l in range(0, n):
        # av += labels[l] if labels[l]*(np.dot(dataset[l], x[:m])+x[m]) < 1 else 0
        # av += labels[l] * max(1-labels[l]*(np.dot(dataset[l], x[:m])+x[m]),0)  # x[m]=b
        av += labels[l] * (1 - labels[l] * (np.dot(dataset[l], x[:m]) + x[m]+x[m+1+l]))
    res.append(av/n)
    return res


def attack_norm_constr(x, e):
    ret = []
    norm = np.dot(x[:m], x[:m]) * (e ** 2)
    for i in range(0, n):
        ret.append(-x[m + 1 + i] ** 2 + norm)
    return ret


def adv_objective(x):
    av = 0.0
    for i in range(0, n):
        av += max(1 + labels[i] * (np.dot(dataset[i], x[:m]) + x[m]), 0)
    return av / n

x0 = np.array([1 for i in range(0, m+n+1)])
con1 = {'type': 'eq', 'fun': class_constr}
options = {'maxiter': 100}
for e in eps_list:
    con2 = {'type': 'ineq', 'fun': attack_norm_constr, 'args': [e]}
    cons = ([con1, con2])
    sol = minimize(adv_objective, x0, bounds=None, constraints=cons, options=options)
    print(sol.nit)
    if not sol.success:
        print(sol.message)
    w = sol.x[:m]
    b = sol.x[m]
    predicted_labels = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
    err = 1 - accuracy_score(labels, predicted_labels)
    maxerrs.append(err)
    colors_h.append((1, 0, 0))
    print('min done e='+str(e))

plt.scatter(eps_list, maxerrs, c=colors_h)
plt.show()

