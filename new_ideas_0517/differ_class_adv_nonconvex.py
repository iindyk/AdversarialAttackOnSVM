import numpy as np
from scipy.optimize import minimize

from random import uniform
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = []
labels = []
colors = []
colors_h = []
n = 50  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0/n  # SVM regularization parameter
attack_size = 0  # random attack size
A = 0
B = 100
eps = 0.1*(B-A)  # upper bound for norm of h
maxit = 50
delta = 1e-2
for i in range(0, n):
    point = []
    for j in range(0, m):
        point.append(uniform(A, B))
    dataset.append(point)
    # change
    if sum([p for p in point]) >= m*(B-A)/2:
        labels.append(1.0)
        colors.append((1, 0, 0))
    else:
        labels.append(-1.0)
        colors.append((0, 0, 1))

# random attack
for i in range(0, attack_size):
    if labels[i] == 1.0:
        labels[i] = -1.0
        colors[i] = (0, 0, 1)
    else:
        labels[i] = 1.0
        colors[i] = (1, 0, 0)
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labels = svc.predict(dataset)
err_orig = 1 - accuracy_score(labels, predicted_labels)
print('err on orig is '+str(err_orig))


def decompose_x(x):
    return np.array(x[:m]), x[m], np.array(x[m+1:m*n+m+1]), \
           np.array(x[m*n+m+1:(m+1)*n+m+1]), np.array(x[(m+1)*n+m+1:(m+2)*n+m+1])   # w, b, h, l, a


def adv_obj(x):
    av = 0.0
    for i in range(0, n):
        av += max(labels[i]*(np.dot(dataset[i], x[:m])+x[m]), -1.0)
    return av


def class_constr_inf_eq(x, w_prev, l_prev):
    ret = []
    w, b, h, l, a = decompose_x(x)
    for j in range(0, m):
        ret.append(w[j] - sum([l_prev[i]*labels[i]*(dataset[i][j]+h[j*n+i]) for i in range(0, n)]))
    ret.append(sum([l[i]*labels[i] for i in range(0, n)]))
    for i in range(0, n):
        hi = [h[j * n + i] for j in range(0, m)]
        ret.append(l[i] - l_prev[i]*a[i] - l_prev[i]*labels[i]*(np.dot(w, dataset[i])+np.dot(w_prev, hi) + b))
        ret.append(l_prev[i]*a[i] - C*a[i])
    return ret


def class_constr_inf_eq_neg(x, w_prev, l_prev):
    ret = []
    w, b, h, l, a = decompose_x(x)
    for j in range(0, m):
        ret.append(w[j] - sum([l_prev[i] * labels[i] * (dataset[i][j] + h[j * n + i]) for i in range(0, n)]))
    ret.append(sum([l[i] * labels[i] for i in range(0, n)]))
    for i in range(0, n):
        hi = [h[j * n + i] for j in range(0, m)]
        ret.append(l[i] - l_prev[i] * a[i] - l_prev[i] * labels[i] * (np.dot(w, dataset[i]) + np.dot(w_prev, hi) + b))
        ret.append(l_prev[i] * a[i] - C * a[i])
    return -1.0*np.array(ret)


def class_constr_inf_ineq(x, w_prev):
    ret = []
    w, b, h, l, a = decompose_x(x)
    for i in range(0, n):
        ret.append(l[i])
        ret.append(C - l[i])
        ret.append(a[i])
        hi = [h[j * n + i] for j in range(0, m)]
        ret.append(labels[i]*(np.dot(w, dataset[i]) + np.dot(w_prev, hi)+b)-1+a[i])
    ret.append(eps*n - np.dot(h, h))
    return ret

l_opt = []
h_opt = []
for i in range(0, n):
    if i in svc.support_:
        #l_opt.append(1.0/n)
        l_opt.append(svc.dual_coef_[0][list(svc.support_).index(i)]/labels[i])
    else:
        l_opt.append(0.0)
x_opt = list(svc.coef_[0]) + list(svc.intercept_)+[0.0 for i in range(0, m*n)] + l_opt + [0.0 for i in range(0, n)]

options = {'maxiter': 1000}
nit = 0
while nit < maxit:
    print('iteration '+str(nit))
    w_p = x_opt[:m]
    l_p = x_opt[m*n+m+1:(m+1)*n+m+1]
    con1 = {'type': 'ineq', 'fun': class_constr_inf_eq, 'args': [w_p, l_p]}
    con2 = {'type': 'ineq', 'fun': class_constr_inf_eq_neg, 'args': [w_p, l_p]}
    con3 = {'type': 'ineq', 'fun': class_constr_inf_ineq, 'args': [w_p]}
    cons = [con1, con2, con3]
    sol = minimize(adv_obj, x_opt, constraints=cons, options=options, method='COBYLA')
    print(sol.success)
    print(sol.message)
    x_opt = sol.x[:]
    w, b, h, l, a = decompose_x(x_opt)
    print(sol.nfev)
    print(sol.maxcv)
    print(w)
    print(b)
    if np.linalg.norm([w[i]-w_p[i] for i in range(0, m)]) < delta \
        and np.linalg.norm([l[i]-l_p[i] for i in range(0, n)]) < delta \
            and sol.success:
        print('maxcv is '+str(sol.maxcv))
        break

dataset_infected = []
print('attack norm is '+str(np.dot(h, h)/n))
for i in range(0, n):
    tmp = []
    for j in range(0, m):
        tmp.append(dataset[i][j]+h[j*n+i])
    dataset_infected.append(tmp)

#print(class_constr_eq_orig(x_opt, dataset_infected))
#print(class_constr_ineq_orig(x_opt, dataset_infected))

svc1 = svm.SVC(kernel='linear', C=C)
svc1.fit(dataset_infected, labels)
predicted_labels_inf_svc = svc1.predict(dataset_infected)
err_inf_svc = 1 - accuracy_score(labels, predicted_labels_inf_svc)
print('err on infected dataset by svc is '+str(err_inf_svc))
predicted_labels_inf_opt = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
err_inf_opt = 1 - accuracy_score(labels, predicted_labels_inf_opt)
print('err on infected dataset by opt is '+str(err_inf_opt))

plt.subplot(221)
plt.title('original')
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.subplot(222)
plt.title('infected')
plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
plt.subplot(223)
step = 1  # step size in the mesh

x_min, x_max = -50, 150
y_min, y_max = -50, 150
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
Z_list = []
for j in range(x_min, x_max):
    for i in range(y_min, y_max):
        Z_list.append(np.sign(xx[i][j]*w[0] + yy[i][j]*w[1]+b))

Z = np.array(Z_list)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.subplot(224)
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()