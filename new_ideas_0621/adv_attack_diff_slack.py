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
n = 10  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0/n  # SVM regularization parameter
attack_size = 2  # random attack size
A = 0
B = 100
eps = 0.1*(B-A)  # upper bound for norm of h
delta = 1e-2  # iteration precision level
maxit = 100  # iteration number limit
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
    return np.array(x[:m]), x[m], np.array(x[m+1:m*n+m+1]), np.array(x[m*n+m+1:(m+1)*n+m+1]), \
           np.array(x[(m+1)*n+m+1:(m+2)*n+m+1]), np.array(x[(m+2)*n+m+1:(m+3)*n+m+1]), \
           np.array(x[(m+3)*n+m+1:(m+4)*n+m+1])   # w, b, h_hat, g, l, xi, psi

bnds = []
for j in range(0, (m+1)*n+m+1):
    bnds.append((-1000, 1000))
for i in range(0, n):
    bnds.append((0, C))
for i in range(0, n):
    bnds.append((0, 1000))


def adv_obj(x):
    return sum(x[(m+3)*n+m+1:(m+4)*n+m+1])


def class_constr_inf_eq(x, w_prev, l_prev):
    ret = []
    w, b, h_hat, g, l, xi, psi = decompose_x(x)
    for j in range(0, m):
        ret.append(w[j] - sum([l[i]*labels[i]*dataset[i][j]+h_hat[j*n+i] for i in range(0, n)]))
    ret.append(sum([l[i]*labels[i] for i in range(0, n)]))
    for i in range(0, n):
        ret.append(l[i] - xi[i] - l_prev[i]*labels[i]*(np.dot(w, dataset[i]) + g[i] + b))
        hi = [h_hat[j*n +i] for j in range(0, m)]
        ret.append(np.dot(w_prev, hi) - l_prev[i]*g[i])
        ret.append(l_prev[i]*xi[i] - C*xi[i])
    return ret


def class_constr_inf_eq_neg(x, w_prev, l_prev):
    ret = []
    w, b, h_hat, g, l, xi, psi = decompose_x(x)
    for j in range(0, m):
        ret.append(w[j] - sum([l[i]*labels[i]*dataset[i][j]+h_hat[j*n+i] for i in range(0, n)]))
    ret.append(sum([l[i]*labels[i] for i in range(0, n)]))
    for i in range(0, n):
        ret.append(l[i] - xi[i] - l_prev[i]*labels[i]*(np.dot(w, dataset[i]) + g[i] + b))
        hi = [h_hat[j*n +i] for j in range(0, m)]
        ret.append(np.dot(w_prev, hi) - l_prev[i]*g[i])
        ret.append(l_prev[i]*xi[i] - C*xi[i])
    return -1.0*np.array(ret)


def class_constr_inf_ineq(x, l_prev, w_prev, ub):
    ret = []
    w, b, h_hat, g, l, xi, psi = decompose_x(x)
    for i in range(0, n):
        ret.append(l[i])
        ret.append(C - l[i])
        ret.append(xi[i])
        ret.append(labels[i]*(np.dot(w, dataset[i])+g[i]+b)-1+xi[i]/C)
        ret.append(-labels[i]*(np.dot(w, dataset[i])+b)+psi[i])
        ret.append((l_prev[i]**2)*n*eps - sum([h_hat[n*j+i]**2 for j in range(0, m)]))  # l==0 => h_hat==0
    ret.append(eps*n - np.dot(g, g))
    ret.append(eps*(C**2)*n - np.dot(h_hat, h_hat))
    #ret.append(1 - np.dot(w, w))
    #ret.append(x[0])
    #ret.append(x[1])
    #ret.append(ub - (w_prev[0] - x[0]) ** 2 - (w_prev[1] - x[1]) ** 2)
    return ret

l_opt = []
h_opt = []
for i in range(0, n):
    if i in svc.support_:
        #l_opt.append(1.0/n)
        l_opt.append(svc.dual_coef_[0][list(svc.support_).index(i)]/labels[i])
    else:
        l_opt.append(0.0)
x_opt = list(svc.coef_[0]) + list(svc.intercept_)+[eps/(n**2) for i in range(0, m*n+n)] + \
        l_opt + [0.0 for i in range(0, 2*n)]
w, b, h_hat, g, l, xi, psi = decompose_x(x_opt)
w_prev = np.array([0.5 for i in range(0, m)])
l_prev = np.array([0.5 for i in range(0, n)])
nit = 0
options = {'maxiter': 100000, 'catol': 1e-2}
ub = 1
con1 = {'type': 'ineq', 'fun': class_constr_inf_eq, 'args': [w_prev, l_prev]}
con2 = {'type': 'ineq', 'fun': class_constr_inf_eq_neg, 'args': [w_prev, l_prev]}
con3 = {'type': 'ineq', 'fun': class_constr_inf_ineq, 'args': [l_prev, w_prev, ub]}
cons = [con1, con2, con3]
#(adv_obj(x_prev) < adv_obj(x_opt) or fl
while nit < maxit:
    x_prev = list(x_opt)
    w_prev = list(w)
    l_prev = list(l)
    print('iteration: ' + str(nit)+'; start: '+str(datetime.datetime.now().time()))
    sol = minimize(adv_obj, x_opt, constraints=cons, options=options, method='COBYLA')
    print('success: '+str(sol.success))
    print('message: '+str(sol.message))
    x_opt = sol.x[:]
    w, b, h_hat, g, l, xi, psi = decompose_x(x_opt)
    print('nfev= '+str(sol.nfev))
    print('maxcv= '+str(sol.maxcv))
    print('w= '+str(w))
    print('b= '+str(b))
    if np.linalg.norm([w[i]-w_prev[i] for i in range(0, m)]) < delta \
        and np.linalg.norm([l[i]-l_prev[i] for i in range(0, n)]) < delta \
            and sol.success:
        break
    if sol.success:
        ub /= 2.0
    nit += 1


dataset_infected = []
h = []
for j in range(0, m):
    for i in range(0, n):
        h.append(0 if abs(l[i]) < delta else h_hat[i+j*n]/l[i])
print('attack norm is '+str(np.dot(h, h)/n))
check = []
for i in range(0, n):
    temp = []
    for j in range(0, m):
        temp.append(dataset[i][j] + h[j * n + i])
    dataset_infected.append(temp)
    check.append(np.dot(w, temp) -np.dot(w, dataset[i]) -g[i])
#print(class_constr_eq_orig(x_opt, dataset_infected))
#print(class_constr_ineq_orig(x_opt, dataset_infected))
print(check)
print(l)

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