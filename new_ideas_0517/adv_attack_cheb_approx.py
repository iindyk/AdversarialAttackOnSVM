import numpy as np
from scipy.optimize import minimize, root

from random import uniform, randint
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
n = 200  # training set size
m = 2  # features
a = 40  # attack size
eps_list = [5*i for i in range(0, 9)]
C = 1.0  # SVM regularization parameter
nsim = 100
# x[:m] = w
# x[m] = b
# x[m+1:n+m+1] = h[:][0]
# x[j*n+m+1:(j+1)*n+m+1] = h[:][j]

for i in range(0, n):
    point = []
    for j in range(0, m):
        point.append(uniform(0, 100))
    dataset.append(point)
    # change
    if sum([p**2 for p in point]) >= 50**2 * m:
        labels.append(1.0)
        colors.append((1, 0, 0))
    else:
        labels.append(-1.0)
        colors.append((0, 0, 1))

# random attack
for i in range(0, a):
    if labels[i] == 1:
        labels[i] = -1
    else:
        labels[i] = 1


def chebyshev_subdiff_inf(x):
    ret = []
    eywxb = 0.0
    eyxi = [0.0 for i in range(0, m)]
    ewxb2 = 0.0
    ewxbxi = [0.0 for i in range(0, m)]
    ey = 0.0
    ewxb = 0.0
    for i in range(0, n):
        eywxb += (labels[i]*(np.dot(x[:m], [dataset[i][j]+x[j*n+i+1+m] for j in range(0, m)])+ x[m]))/n
        ewxb2 += ((np.dot(x[:m], [dataset[i][j]+x[j*n+i+1+m] for j in range(0, m)])+x[m])**2)/n
        ey += (labels[i])/n
        ewxb += (np.dot(x[:m], [dataset[i][j]+x[j*n+i+1+m] for j in range(0, m)])+x[m])/n
        for j in range(0, m):
            eyxi[j] += (labels[i]*(dataset[i][j]+x[j*n+i+1+m]))/n
            ewxbxi[j] += ((np.dot(x[:m], [dataset[i][k]+x[k*n+i+1+m] for k in range(0, m)])+x[m])*(dataset[i][j]+x[j*n+i+1+m]))/n
    for j in range(0, m):
        ret.append(-2*eywxb*eyxi[j]*ewxb2 + 2*ewxbxi[j])
    ret.append(-2*eywxb*ey*ewxb2 + 2*ewxb)
    return ret


def wsign_constr(x):
    neywxb = 0.0
    for i in range(0, m):
        neywxb += (labels[i]*(np.dot(x[:m], [dataset[i][j]+x[j*n+i+1+m] for j in range(0, m)]) + x[m]))
    return -neywxb

'''x0 = np.array([1.0 for i in range(0, m+1)])
sol = root(chebyshev_subdiff, x0)
print(sol.success)
print(sol.message)
print(wsign_constr(sol.x))
if wsign_constr(sol.x) >= 0:
    w = sol.x[:m]
    b = sol.x[m]
else:
    w = -sol.x[:m]
    b = -sol.x[m]
print(sol.x[:m])
print(w)
predicted_labelsS = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
accS = accuracy_score(labels, predicted_labelsS)
errS = 1 - accS
print("Chebyshev's svm error "+str(errS))
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labelsC = svc.predict(dataset)
accC = accuracy_score(labels, predicted_labelsC)
errC = 1 - accC
print("c-svm error "+str(errC))'''


def adv_obj(x):
    av = 0.0
    for i in range(0, n):
        av += max(labels[i]*(np.dot(dataset[i], x[:m])+x[m]), -1.0)
    return av/n


def attack_norm_constr(x, eps):
    ret = 0.0
    for i in range(0, n):
            for j in range(0, m):
                ret += x[j*n+i+1+m]**2
    return -(ret/n)+eps**2


con1 = {'type': 'eq', 'fun': chebyshev_subdiff_inf}
con2 = {'type': 'ineq', 'fun': wsign_constr}
x0 = np.array([1.0 for i in range(0, m+1+n*m)])
errs = []
colors_h = []
h20 = []
ds_inf20 = []
for eps in eps_list:
    con3 = {'type': 'ineq', 'fun': attack_norm_constr, 'args': [eps]}
    cons = [con1, con2, con3]
    sol = minimize(adv_obj, x0, constraints=cons)
    print(sol.success)
    print(sol.message)
    w = sol.x[:m]
    b = sol.x[m]
    h = sol.x[m+1:]
    dataset_infected = []
    for i in range(0, n):
        temp = []
        for j in range(0, m):
            temp.append(dataset[i][j] + h[j * n + i])
        dataset_infected.append(temp)
    if eps == 20:
        h20 = list(h)
        ds_inf20 = list(dataset_infected)
    #predicted_labelsS_inf = np.sign([np.dot(dataset_infected[i], w)+b for i in range(0, n)])
    #accS_inf = accuracy_score(labels, predicted_labelsS_inf)
    #errS_inf = 1 - accS_inf
    #print("Chebyshev's svm error "+str(errS_inf))
    svc = svm.SVC(kernel='linear', C=C).fit(dataset_infected, labels)
    errs.append(1 - accuracy_score(labels, svc.predict(dataset_infected)))
    colors_h.append((1, 0, 0))

plt.subplot(221)
plt.title('original dataset')
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.subplot(222)
plt.title('infected dataset (chebyshev), eps = 20')
plt.scatter([float(i[0]) for i in ds_inf20], [float(i[1]) for i in ds_inf20], c=colors, cmap=plt.cm.coolwarm)
h_list = []
h = []
for eps in eps_list:
    h_list = []
    maxerr = 0.0
    for k in range(0, nsim):
        h = []
        dataset_infected = []
        for i in range(0, n):
            fi = uniform(0, 2*np.pi)
            h.append([eps*np.cos(fi), eps*np.sin(fi)])
            dataset_infected.append([dataset[i][0]+eps*np.cos(fi), dataset[i][1] + eps*np.sin(fi)])
        h_list.append(list(h))
        svc = svm.SVC(kernel='linear', C=C).fit(dataset_infected, labels)
        if maxerr < 1 - accuracy_score(labels, svc.predict(dataset_infected)):
            maxerr = 1 - accuracy_score(labels, svc.predict(dataset_infected))
            h_maxerr = list(h)
    errs.append(maxerr)
    colors_h.append((0, 1, 0))

plt.subplot(223)
plt.title('optimization attacks vs best random(out of 100 sim.)')
plt_opt = plt.scatter(errs[:9], eps_list, c=colors_h[:9])
plt_sim = plt.scatter(errs[9:], eps_list, c=colors_h[9:])
plt.legend([plt_opt, plt_sim], ['optimization', 'simulation'])

