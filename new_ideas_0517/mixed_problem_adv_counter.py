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
m = 2  # features
C = 1.0  # SVM regularization parameter
a = 60  # random attack size
alpha = 1.0 # classifier objective weight
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
for i in range(0, a):
    if labels[i] == 1:
        labels[i] = -1
    else:
        labels[i] = 1


# x[:m] - w
# x[m] - b
# x[m+1] - w*h
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
        av += max(1-labels[i]*(np.dot(dataset[i], x[:m])+x[m]+x[m+1]), 0)
    return np.dot(x[:m], x[:m])/2.0 + C*av


x0 = np.array([1 for i in range(0, m+2)])
solution = minimize(objective, x0, bounds=None)
print(solution.x)
print(solution.message)
print(solution.nit)