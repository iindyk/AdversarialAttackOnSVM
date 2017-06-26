import datetime
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
n = 20  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0  # SVM regularization parameter
flip_size = 3  # random attack size
A = 0
B = 100
eps = 1.0*(B-A)  # upper bound for (norm of h)**2
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
for i in range(0, flip_size):
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
    return x[:m], x[m], x[m+1:m*n+m+1], \
           x[m*n+m+1:(m+1)*n+m+1], x[(m+1)*n+m+1:(m+2)*n+m+1]   # w, b, h, l, a


def adv_obj(x):
    av = 0.0
    for i in range(0, n):
        av += max(labels[i]*(np.dot(dataset[i], x[:m])+x[m]), -1.0)
        #av += 1 if labels[i]*(np.dot(dataset[i], x[:m])+x[m]) > 0 else 0
        #av += labels[i]*(np.dot(dataset[i], x[:m])+x[m])
    return av/n


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
    return np.array(ret)


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
        ret.append(labels[i]*(np.dot(w, dataset[i]) + np.dot(w_prev, [h[j * n + i] for j in range(0, m)])+b)-1+a[i])
    ret.append(eps*n - np.dot(h, h))
    #ret.append(1 - np.dot(w, w))
    #ret.append(1 - err_orig - adv_obj(x))
    return np.array(ret)

l_opt = []
h_opt = []
for i in range(0, n):
    if i in svc.support_:
        #l_opt.append(1.0/n)
        l_opt.append(svc.dual_coef_[0][list(svc.support_).index(i)]/labels[i])
    else:
        l_opt.append(0.0)
x_opt = list(svc.coef_[0]) + list(svc.intercept_)+[np.sqrt(eps)/(m*n) for i in range(0, m*n)] + l_opt + [0.0 for i in range(0, n)]

options = {'maxiter': 100000}
nit = 0
while nit < maxit:
    print('iteration '+str(nit)+'; start: '+str(datetime.datetime.now().time()))
    w_p = x_opt[:m]
    l_p = x_opt[m*n+m+1:(m+1)*n+m+1]
    con1 = {'type': 'ineq', 'fun': class_constr_inf_eq, 'args': [w_p, l_p]}
    con2 = {'type': 'ineq', 'fun': class_constr_inf_eq_neg, 'args': [w_p, l_p]}
    con3 = {'type': 'ineq', 'fun': class_constr_inf_ineq, 'args': [w_p]}
    cons = [con1, con2, con3]
    sol = minimize(adv_obj, x_opt, constraints=cons, options=options, method='COBYLA')
    print('success: '+str(sol.success))
    print('message: '+str(sol.message))
    x_opt = sol.x[:]
    w, b, h, l, a = decompose_x(x_opt)
    print('nfev= '+str(sol.nfev))
    #print('maxcv= '+str(sol.maxcv))
    print('w= '+str(w))
    print('b= '+str(b))
    x_p = list(w_p) + [b] + list(h) + list(l_p) + list(a)
    if adv_obj(x_p) <= sol.fun+delta and class_constr_inf_eq(x_p, w_p, l_p).all() >= -delta \
        and class_constr_inf_eq_neg(x_p, w_p, l_p).all() >= -delta \
            and class_constr_inf_ineq(x_p, w_p).all() >= -delta \
            and sol.success and np.dot(h, h) / n > eps - 0.1:
        break
    #if np.linalg.norm([w[i]-w_p[i] for i in range(0, m)]) < delta \
     #   and np.linalg.norm([l[i]-l_p[i] for i in range(0, n)]) < delta \
      #      and sol.success and np.dot(h, h)/n > eps - 0.1:
       # break
    nit += 1

dataset_infected = []
print('attack norm= '+str(np.dot(h, h)/n))
print('objective value= '+str(sol.fun))
for i in range(0, n):
    tmp = []
    for j in range(0, m):
        tmp.append(dataset[i][j]+h[j*n+i])
    dataset_infected.append(tmp)

#print(class_constr_eq_orig(x_opt, dataset_infected))
#print(class_constr_ineq_orig(x_opt, dataset_infected))

svc1 = svm.SVC(kernel='linear', C=C)
svc1.fit(dataset_infected, labels)
predicted_labels_inf_svc = svc1.predict(dataset)
err_inf_svc = 1 - accuracy_score(labels, predicted_labels_inf_svc)
print('err on infected dataset by svc is '+str(err_inf_svc))
predicted_labels_inf_opt = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
err_inf_opt = 1 - accuracy_score(labels, predicted_labels_inf_opt)
print('err on infected dataset by opt is '+str(err_inf_opt))

plt.subplot(321)
plt.title('original')
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.subplot(322)
plt.title('infected')
plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
plt.subplot(323)
step = 1  # step size in the mesh

x_min, x_max = int(A-2*(eps/m)**0.5), int(B+2*(eps/m)**0.5)
y_min, y_max = int(A-2*(eps/m)**0.5), int(B+2*(eps/m)**0.5)
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
Z = np.sign([i[0]*w[0]+i[1]*w[1]+b for i in np.c_[xx.ravel(), yy.ravel()]])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
plt.title('opt on inf data')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.subplot(324)
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
plt.title('inf svc on inf data')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.subplot(326)
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.title('inf svc on orig data')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.subplot(325)
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.title('orig svc on orig data')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()