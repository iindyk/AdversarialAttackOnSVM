import numpy as np

from random import uniform
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
n = 400  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C_list = [i for i in range(1, 50)]  # SVM regularization parameter
a = 2  # random attack size
A = 10
B = 110
eps_list = [(i/100.0)*(B-A) for i in range(24, 25)]  # upper bound for norm of h
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
for C in C_list:
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

#print(labels)
#print(errs)
#print(h_d_list)
#print(infected_dataset)
#print(dataset)
plt.scatter(C_list, maxerrs)
plt.show()

