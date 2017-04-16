import numpy as np
from cvxopt import matrix, solvers

from random import uniform
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
n = 600
la = 0.8

for i in range(0, n):
    x1 = uniform(0, 200)
    x2 = uniform(0, 200)
    dataset.append([x1, x2])
    if x2 >= x1 - 10:
        labels.append(1.0)
        colors.append((1, 0, 0))
    else:
        labels.append(-1.0)
        colors.append((0, 0, 1))

true_labels = labels[:]

q_list = []
g_list = [[0 for j in range(0, 2*n)] for k in range(0, n)]
h_list = [0 for l in range(0, 2*n)]

for i in range(0, n):
    q_list.append([])
    for j in range(0, n):
        if i == j:
            q_list[i].append(np.dot(dataset[i], dataset[i]) * (labels[i] ** 2))
            g_list[i][i] = -1.0
            g_list[i][i+n] = 1.0
            h_list[i+n] = 1/(2*la*n)
        else:
            q_list[i].append(0.5*np.dot(dataset[i], dataset[j])*labels[i]*labels[j])

Q = 2*matrix(q_list)
p = matrix([-1.0 for i in range(0, n)])
G = matrix(g_list)
h = matrix(h_list)
A = np.transpose(matrix(labels))
b = matrix(0.0)

sol = solvers.qp(matrix(Q), p, matrix(G), h, matrix(A), b)
c = sol['x']
w = [0, 0]
for i in range(0, n):
    w[0] += c[i] * labels[i] * dataset[i][0]
    w[1] += c[i] * labels[i] * dataset[i][1]
# need to find b
mindist = 10000
midpoint = [0, 0]# magic
for i in range(0, n):
    for j in range(0, n):
        if mindist > (dataset[i][0]-dataset[j][0])**2+(dataset[i][1]-dataset[j][1])**2 and labels[i] != labels[j]:
            mindist = (dataset[i][0]-dataset[j][0])**2+(dataset[i][1]-dataset[j][1])**2
            midpoint = [(dataset[i][0]+dataset[j][0])/2, (dataset[i][1]+dataset[j][1])/2]

print(midpoint)
b = np.dot(midpoint, w)


h = 1  # step size in the mesh

x_min, x_max = 0, 200
y_min, y_max = 0, 200
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z_list = []
for i in range(x_min, x_max):
    for j in range(y_min, y_max):
        Z_list.append(np.sign(xx[i][j]*w[0] + yy[i][j]*w[1]-b))

predicted_labels = np.sign([np.dot(dataset[i], w)-b for i in range(0, n)])
acc = accuracy_score(true_labels, predicted_labels)
err = 1 - acc

# Put the result into a color plot

Z = np.array(Z_list)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('linear svm, err=' + str(err))

plt.show()
