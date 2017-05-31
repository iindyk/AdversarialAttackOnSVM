import numpy as np
from scipy.optimize import minimize, root

from random import uniform, randint
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
n = 600  # training set size
m = 15  # features
a = 40  # attack size
eps = 10
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
    else:
        labels.append(-1.0)

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


def attack_norm_constr(x):
    ret = 0.0
    for i in range(0, n):
            for j in range(0, m):
                ret += x[j*n+i+1+m]**2
    return -(ret/n)+eps**2


con1 = {'type': 'eq', 'fun': chebyshev_subdiff_inf}
con2 = {'type': 'ineq', 'fun': wsign_constr}
con3 = {'type': 'ineq', 'fun': attack_norm_constr}
cons = [con1, con2, con3]
x0 = np.array([1.0 for i in range(0, m+1+n*m)])
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
predicted_labelsS_inf = np.sign([np.dot(dataset_infected[i], w)+b for i in range(0, n)])
accS_inf = accuracy_score(labels, predicted_labelsS_inf)
errS_inf = 1 - accS_inf
print("Chebyshev's svm error "+str(errS_inf))
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(dataset_infected, labels)
predicted_labelsC_inf = svc.predict(dataset_infected)
accC_inf = accuracy_score(labels, predicted_labelsC_inf)
errC_inf = 1 - accC_inf
print("c-svm error "+str(errC_inf))
