import numpy as np
from scipy.optimize import minimize

from random import uniform, randint
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
n = 600
m = 2
la = 0.8
C = 1.0  # SVM regularization parameter

for i in range(0, n):
    x1 = uniform(0, 200)
    x2 = uniform(10, 190)
    dataset.append([x1, x2])
    # change
    if 3*x1-4*x2+10 >= 0:
        labels.append(1.0)
        colors.append((1, 0, 0))
    else:
        labels.append(-1.0)
        colors.append((0, 0, 1))

true_labels = labels[:]

# random attack
for i in range(0, 60):
    nnn = randint(0, 200)
    if labels[nnn] == 1:
        labels[nnn] = -1
        colors[nnn] = (0, 0, 1)
    else:
        labels[nnn] = 1
        colors[nnn] = (1, 0, 0)

# ww[2]=b


def objective(x):
    av = 0.0
    for l in range(0, n):
        av += labels[i] * max(1-labels[l]*(np.dot(dataset[l], x[:m])+x[m]),0)  # x[m]=b
    return abs(av)

def obj1(x):
    return np.dot(x, x)


def constraint1(x):
    summ = 0.0
    for s in range(0, n):
        summ += labels[s]*dataset[s][0]*max(1-labels[s]*(np.dot(dataset[s], x[:m])+x[m]),0)
    return x[0]-C*(1.0/n)*summ


def constraint2(x):
    summ = 0.0
    for s in range(0, n):
        summ += labels[s]*dataset[s][1]*max(1-labels[s]*(np.dot(dataset[s], x[:m])+x[m]),0)
    return x[1]-C*(1.0/n)*summ
#
# optimize
b = (-1, 1)
c = (-200, 200)
x0 = np.array([1, -2, 10])
bnds = (b, b, c)
con1 = {'type': 'eq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
con3 = {'type': 'eq', 'fun': objective}
cons = ([con1, con2, con3])
solution = minimize(obj1, x0, bounds=None, constraints=cons)
h = 1  # step size in the mesh

w = [0, 0]
w[0] = solution.x[0]
w[1] = solution.x[1]
b = solution.x[2]

x_min, x_max = 0, 200
y_min, y_max = 0, 200
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z_list = []
for i in range(x_min, x_max):
    for j in range(y_min, y_max):
        Z_list.append(np.sign(xx[i][j]*w[0] + yy[i][j]*w[1]+b))

predicted_labels = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
acc = accuracy_score(labels, predicted_labels)
err = 1 - acc

# Put the result into a color plot

Z = np.array(Z_list)
Z = Z.reshape(xx.shape)
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('stochastic svm, err=' + str(err))



h = 1  # step size in the mesh
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
x_min, x_max = 0, 200
y_min, y_max = 0, 200
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

predicted_labels = svc.predict(dataset)
acc = accuracy_score(labels, predicted_labels)
err = 1 - acc
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.subplot(122)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.xlabel('feature1')
plt.ylabel('feature2')
#plt.xlim(xx.min(), xx.max())
#plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('linear(soft with C=1) svm, err=' + str(err))

plt.show()
