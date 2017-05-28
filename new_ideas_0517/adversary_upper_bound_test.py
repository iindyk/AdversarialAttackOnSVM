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
n = 100  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0/n  # SVM regularization parameter
a = 10  # random attack size
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


def attack_norm_constr(x, w):
    ret = 0.0
    for i in range(0, n):
        ret += x[m+1+i]**2
    return -(ret/n)+np.dot(w, w)*(eps**2)


def adv_constr(x, p, q):
    ret = []
    ret.append(1000*q - adv_obj(x))
    ret.append(adv_obj(x) - 1000*p)
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
w_svc = np.array([0.5 for i in range(0, m)])
w_svc[0] = 0.0
b_svc = 0
w_opt = np.array([1.0 for i in range(0, m)])
b_opt = 1
x_opt = np.array([1.0 for i in range(0, m+n+1)])
u = 0.0
v = 1.0
options = {'maxiter': 100}
while abs(w_svc[0]/w_opt[0] - w_svc[1]/w_opt[1]) > delta and nit < maxit:
    print('iteration '+str(nit))
    con1 = {'type': 'ineq', 'fun': attack_norm_constr, 'args': [w_opt]}
    con2 = {'type': 'ineq', 'fun': adv_constr, 'args': [u, v]}
    cons = [con1, con2]
    sol = minimize(class_obj_inf, x_opt, method='SLSQP', bounds=None, options=options, constraints=cons)
    x_opt = list(sol.x)
    w_opt = sol.x[:m]
    b_opt = sol.x[m]
    g_opt = sol.x[m+1:]
    print(sol.message)
    print(sol.nit)
    print('maxcv is ' + str(attack_norm_constr(x_opt, w_opt))+'  and  '+str(min(adv_constr(x_opt, u, v))))
    if not sol.success:
        print('u = '+str(u)+' v = '+str(v)+' no sol')
        tmp = u
        u = v
        v = min(2*v - tmp, 1.0)
    else:
        print('u = ' + str(u) + ' v = ' + str(v) + ' exist sol')
        v = (u+v)/2
    sol.clear()
    # restoring h
    h0 = np.array([1 for i in range(0, 2 * n)])
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
print(w_opt)
print(b_opt)
print(w_svc)
print(b_svc)
print(err_orig)
print('err on w from opt '+str(err_inf))
pr_lb = svc.predict(dataset)
err_infc = 1 - accuracy_score(labels, pr_lb)
print('err of csvm on infected' + str(err_infc))
####################
nsim = 100
eps_list = [eps]
h_list = []
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
print('mc maxerr '+str(maxerrs))
####################
plt.subplot(221)
plt.title('original')
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.subplot(222)
plt.title('infected')
plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
plt.show()