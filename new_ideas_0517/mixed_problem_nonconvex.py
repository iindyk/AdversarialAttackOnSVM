import numpy as np
from scipy.optimize import minimize

from random import uniform, randint
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
n = 200  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0  # SVM regularization parameter
a = 60  # random attack size
alpha = 0  # classifier objective weight
A = 10
B = 110
val_cl = []
val_ad = []
eps = 0.1*(B-A)  # upper bound for norm of h
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

svc1 = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labelsCO = svc1.predict(dataset)
errCO = 1 - accuracy_score(labels, predicted_labelsCO)


# x[:m] - w
# x[m] - b
# x[m+1:m+n+1] - w*h(\omega)
def objective(x):
    return adv_obj(x) + alpha*class_obj(x)


def adv_obj(x):
    av = 0.0
    for i in range(0, n):
        av += max(1+labels[i]*(np.dot(dataset[i], x[:m])+x[m]), 0)
    return av/n


def class_obj(x):
    av = 0.0
    for i in range(0, n):
        av += max(1-labels[i]*(np.dot(dataset[i], x[:m])+x[m]+x[m+1+i]), 0)
    return np.dot(x[:m], x[:m])/2.0 + C*av/n


def constraint1(x):
    ret = []
    norm = np.dot(x[:m], x[:m])*(eps**2)
    for i in range(0, n):
        ret.append(-x[m+1+i]**2 + norm)
    return ret


def constraint2(x):
    return - adv_obj(x) - errCO


def class_constr(x):
    ret = []
    w0 = np.array([1.0 for i in range(0, m+1)])
    sol = minimize(class_obj_inf, w0, args=[x[m+1:]])
    for i in range(0, m+1):
        ret.append(x[i] - sol.x[i])
    return ret


def class_obj_inf(x, gg):
    av = 0.0
    for i in range(0, n):
        av += max(1 - labels[i] * (np.dot(dataset[i], x[:m]) + x[m] + gg[i]), 0)
    return np.dot(x[:m], x[:m]) / 2.0 + C *av

x0 = np.array([1 for i in range(0, m+n+1)])
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'eq', 'fun': class_constr}
cons = ([con1, con2])
options = {'maxiter': 400}
solution = minimize(objective, x0, bounds=None, method='SLSQP', constraints=cons, options=options)
print(solution.success)
print(solution.message)
print(solution.nit)
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
        ret.append(h[i]*w[0]+h[n+i]*w[1]-g[i])
    return ret

h0 = np.array([0.1 for i in range(0, 2*n)])
con_h = {'type': 'eq', 'fun': constr_h}
cons_h = ([con_h])
solution_h = minimize(obj_h, h0, bounds=None, constraints=cons_h)
print(solution_h.success)
print(solution_h.message)
print(solution_h.nit)
h = solution_h.x

dataset_infected = []
for i in range(0, n):
    temp = []
    for j in range(0, m):
        temp.append(dataset[i][j]+h[j*n+i])
    dataset_infected.append(temp)

predicted_labelsS = np.sign([np.dot(dataset_infected[i], w)+b for i in range(0, n)])
errS = 1 - accuracy_score(labels, predicted_labelsS)
print("mixed svm error "+str(errS))

svc = svm.SVC(kernel='linear', C=C).fit(dataset_infected, labels)
print(svc.coef_)
predicted_labelsC = svc.predict(dataset_infected)
errC = 1 - accuracy_score(labels, predicted_labelsC)
print("c-svm error " + str(errC))

#  plots
if m == 2:
    x_min, x_max = A-10, B+10
    y_min, y_max = A-10, B+10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1.0),
                         np.arange(y_min, y_max, 1.0))
    Z_list = []
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            Z_list.append(np.sign(xx[i][j]*w[0] + yy[i][j]*w[1]+b))
    Z = np.array(Z_list)
    Z = Z.reshape(xx.shape)
    plt.subplot(221)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('mixed svm on infected dataset, err=' + str(errS))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(222)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # Plot also the training points
    plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('linear(soft with C=1) svm on infected dataset, err=' + str(errC))
    plt.subplot(223)
    Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors,
                cmap=plt.cm.coolwarm)
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('linear(soft with C=1) svm on orig dataset, err=' + str(errCO))
    plt.suptitle('mix of classifier\'s and adversary\'s objectives test, weight of obj_cl = ' + str(alpha))
    plt.show()