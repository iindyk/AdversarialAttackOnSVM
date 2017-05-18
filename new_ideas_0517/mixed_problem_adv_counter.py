import numpy as np
from scipy.optimize import minimize

from random import uniform, randint
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
n = 200  # training set size
m = 2  # features
C = 1.0  # SVM regularization parameter
a = 20  # random attack size
alpha = 1.0  # classifier objective weight
eps = 0.1  # upper bound for norm of h
for i in range(0, n):
    point = []
    for j in range(0, m):
        point.append(uniform(0, 100))
    dataset.append(point)
    # change
    if sum([p**2 for p in point]) >= 5000:
        labels.append(1.0)
    else:
        labels.append(-1.0)

# random attack
for i in range(0, a):
    if labels[i] == 1:
        labels[i] = -1
    else:
        labels[i] = 1


# x[:m] - w
# x[m] - b
# x[m+1:m+n+1] - w*h(\omega)
def objective(x):
    return adv_obj(x) + alpha*class_obj(x)


def adv_obj(x):
    av = 0.0
    for i in range(0, n):
        av += max(labels[i]*(np.dot(dataset[i], x[:m])+x[m]), -1)
    return av/n


def class_obj(x):
    av = 0.0
    for i in range(0, n):
        av += max(1-labels[i]*(np.dot(dataset[i], x[:m])+x[m]+x[m+1+i]), 0)
    return np.dot(x[:m], x[:m])/2.0 + C*av/n


def constraint(x):
    ret = []
    for i in range(0, n):
        ret.append(np.dot(x[:m], x[:m])*(eps**2)-x[m+1+i]**2)
    return ret


x0 = np.array([1 for i in range(0, m+n+1)])
con1 = {'type': 'ineq', 'fun': constraint}
cons = ([con1])
solution = minimize(objective, x0, bounds=None, method='SLSQP', constraints=cons)
print(solution.message)
print(solution.nit)
print(constraint(solution.x))
w = solution.x[:m]
b = solution.x[m]
g = solution.x[m+1:m+n+1]


#  restoring h
#  h[:n] - h1s, h[n:2*n] - h2s
def obj_h(h):
    return np.dot(h, h)


def constr_h(h):
    ret = []
    for i in range(0, n):
        ret.append(h[i]*w[1]+h[n+i]*w[2]-g[m+1+i])
    return ret

h0 = np.array([0.1 for i in range(0, 2*n)])
con_h = {'type': 'eq', 'fun': constr_h}
cons_h = ([con_h])
solution_h = minimize(obj_h, h0, bounds=None, constraints=cons_h)
print(solution_h.success)
print(solution_h.nit)
print(solution_h.message)
print(solution_h.x)




##############
'''predicted_labelsS = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
errS = 1 - accuracy_score(labels, predicted_labelsS)
print("approximation svm error "+str(errS))

svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labelsC = svc.predict(dataset)
errC = 1 - accuracy_score(labels, predicted_labelsC)
print("c-svm error " + str(errC))'''