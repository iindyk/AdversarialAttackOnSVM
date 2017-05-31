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


def chebyshev_subdiff(x):
    ret = []
    eywxb = 0.0
    eyxi = [0.0 for i in range(0, m)]
    ewxb2 = 0.0
    ewxbxi = [0.0 for i in range(0, m)]
    ey = 0.0
    ewxb = 0.0
    for i in range(0, n):
        eywxb += (labels[i]*(np.dot(x[:m], dataset[i]) + x[m]))/n
        ewxb2 += ((np.dot(x[:m], dataset[i])+x[m])**2)/n
        ey += (labels[i])/n
        ewxb += (np.dot(x[:m], dataset[i])+x[m])/n
        for j in range(0, m):
            eyxi[j] += (labels[i]*dataset[i][j])/n
            ewxbxi[j] += ((np.dot(x[:m], dataset[i])+x[m])*dataset[i][j])/n
    for j in range(0, m):
        ret.append(-2*eywxb*eyxi[j]*ewxb2 + 2*ewxbxi[j])
    ret.append(-2*eywxb*ey*ewxb2 + 2*ewxb)
    return ret

x0 = [1.0 for i in range(0, m+1)]
sol = root(chebyshev_subdiff, x0)
w = sol.x[:m]
b = sol.x[m]
predicted_labelsS = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
accS = accuracy_score(labels, predicted_labelsS)
errS = 1 - accS
print("Chebyshev's svm error "+str(errS))
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labelsC = svc.predict(dataset)
accC = accuracy_score(labels, predicted_labelsC)
errC = 1 - accC
print("c-svm error "+str(errC))