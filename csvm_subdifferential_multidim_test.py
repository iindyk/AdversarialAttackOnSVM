import numpy as np
from scipy.optimize import root

from random import uniform, randint
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
n = 600  # training set size
m = 15  # features
C = 1.0  # SVM regularization parameter

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


def F(x):
    res = []
    for i in range(0, m):
        summ = 0.0
        for j in range(0, n):
            # summ += labels[j] * dataset[j][i] * (1 if labels[j]*(np.dot(dataset[j], x[:m])+x[m]) < 1 else 0)
            # summ += labels[j]*dataset[j][i]*max(1-labels[j]*(np.dot(dataset[j], x[:m])+x[m]),0)
            summ += labels[j] * dataset[j][i] * (1 - labels[j] * (np.dot(dataset[j], x[:m]) + x[m]))
        res.append(x[i] - C*(1.0/n)*summ)
    av = 0.0
    for l in range(0, n):
        # av += labels[l] if labels[l]*(np.dot(dataset[l], x[:m])+x[m]) < 1 else 0
        # av += labels[l] * max(1-labels[l]*(np.dot(dataset[l], x[:m])+x[m]),0)  # x[m]=b
        av += labels[l] * (1 - labels[l] * (np.dot(dataset[l], x[:m]) + x[m]))
    res.append(av/n)
    return res

# optimize
x0 = np.asarray([1 for i in range(0, m+1)])
solution = root(F, x0, tol=1e-13)
print(solution.fun)
print(solution.success)
print(solution.message)
h = 1  # step size in the mesh

w = solution.x[0:m]
b = solution.x[m]

predicted_labelsS = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
accS = accuracy_score(labels, predicted_labelsS)

errS = 1 - accS
print("stoch svm error "+str(errS))


svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)

predicted_labelsC = svc.predict(dataset)
accC = accuracy_score(labels, predicted_labelsC)
errC = 1 - accC

erS = 0
erC = 0
diff = 0
for k in range(0, n):
    if predicted_labelsS[k] != labels[k]:
        erS += 1
    if predicted_labelsC[k] != labels[k]:
        erC += 1
    if predicted_labelsS[k] != predicted_labelsC[k]:
        diff += 1

print("c-svm error " + str(errC))
print("st svm errors " + str(erS))
print("c-svm errors " + str(erC))
print("diff " + str(diff))
