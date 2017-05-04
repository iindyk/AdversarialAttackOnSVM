import numpy as np
from scipy.optimize import minimize

from random import uniform, randint
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
n = 600  # training set size
m = 15  # features

for i in range(0, n):
    point = []
    for j in range(0, m):
        point.append(uniform(0, 100))
    dataset.append(point)
    # change
    if sum([p**2 for p in point]) >= 37500:
        labels.append(1.0)
    else:
        labels.append(-1.0)

# random attack
for i in range(0, 60):
    if labels[i] == 1:
        labels[i] = -1
    else:
        labels[i] = 1


def objective_cheb(x):
    av = 0.0
    for l in range(0, len(dataset)):
        temp = 0.0
        for k in range(0, m):
            temp += dataset[l][k]*x[k]
        av += (labels[l]*(temp + x[m]))  # x[m]=b
    return -av/len(dataset)


def constr_cheb(x):
    dev = 0.0
    for l in range(0, len(dataset)):
        temp = 0.0
        for k in range(0, m):
            temp += dataset[l][k] * x[k]
        dev += ((labels[l] * (temp + x[m])) ** 2)  # x[m]=b
    return dev - 1

# optimize
con1 = {'type': 'eq', 'fun': constr_cheb}
cons = ([con1])
options = {'maxiter': 200}
x0 = np.asarray([1.0 for p in range(0, m+1)])
solution = minimize(objective_cheb, x0, bounds=None, constraints=cons, options=options)
h = 1  # step size in the mesh

print(solution.fun)
print(solution.success)
print(solution.message)
print(solution.nit)

w = solution.x[0:m]
b = solution.x[m]

predicted_labelsS = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
accS = accuracy_score(labels, predicted_labelsS)

errS = 1 - accS
print("Chebyshev's svm error "+str(errS))
C = 1.0  # SVM regularization parameter

svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)

predicted_labelsC = svc.predict(dataset)
accC = accuracy_score(labels, predicted_labelsC)
errC = 1 - accC

erS = 0
erC = 0
diff = 0
for k in range(0, n):
    if predicted_labelsS[k]!=labels[k]:
        erS += 1
    if predicted_labelsC[k]!=labels[k]:
        erC += 1
    if predicted_labelsS[k] != predicted_labelsC[k]:
        diff += 1

print("c-svm error "+str(errC))
print("st svm errors "+str(erS))
print("c-svm errors "+ str(erC))
print("diff "+str(diff))
