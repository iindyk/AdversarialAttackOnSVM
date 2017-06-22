import numpy as np
from scipy.optimize import minimize

from random import uniform
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


dataset = []
labels = []
colors = []
n = 20  # training set size (must be larger than m to avoid fuck up)
m = 2  # features
C = 1.0/n  # SVM regularization parameter
flip_size = 2  # random attack size
A = 0
B = 100
eps = 0.1*(B-A)  # upper bound for (norm of h)**2
maxit = 50
delta = 1e-5
alpha = 1  # classifier objective weight
###############
#  x[:m]=w; x[m]=b; x[m+1:m+n+1]=g; x[m+n+1:m+2*n+1]=psi; x[m+2*n+1:m+3*n+1]=xi
###############
for i in range(0, n):
    point = []
    for j in range(0, m):
        point.append(uniform(A, B))
    # change
    if sum([p for p in point]) >= m*(B-A)/2:
        labels.append(1.0)
        colors.append((1, 0, 0))
        #point[0] += 2
    else:
        labels.append(-1.0)
        colors.append((0, 0, 1))
        #point[0] -= 2
    dataset.append(point)
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


def objective(x):
    return adversary_obj(x) + alpha*classifier_obj(x)


def adversary_obj(x):
    return -sum(x[m+n+1:m+2*n+1])


def classifier_obj(x):
    return C*sum(x[m+2*n+1:m+3*n+1]) + np.dot(x[:m], x[:m])


def class_constr_inf_ineq(x, w_prev, ub):
    ret = []
    for i in range(0, n):
        ret.append(labels[i]*(np.dot(x[:m], dataset[i]) + x[m+1+i]+x[m]) - 1.0 + x[m+2*n+1+i])
        ret.append(-labels[i]*(np.dot(x[:m], dataset[i]) + x[m]) + 1.0 - x[m+n+1+i])
    ret.append(eps*np.dot(w_prev, w_prev)*n - np.dot(x[m+1:m+n+1], x[m+1:m+n+1]))
    ret.append(np.dot(x[:m], w_prev) - 0.01)
    ret.append(ub - (w_prev[0] - x[0])**2 - (w_prev[1] - x[1])**2)
    return np.array(ret)

bnds = [(0.0, 10.0), (0.0, 10.0), (-100.0, 100.0)]
for i in range(0, n):
    bnds.append((-n*eps, n*eps))
for i in range(0, 2*n):
    bnds.append((0, 1e10))

x_opt = [0.05 for i in range(0, m+1+3*n)]
options = {'maxiter': 20000, 'disp': True}
nit = 0
ub = 1
while nit < maxit:
    print('iteration ' + str(nit))
    w_p = x_opt[:m]
    con = {'type': 'ineq', 'fun': class_constr_inf_ineq, 'args': [w_p, ub]}
    cons = [con]
    sol = minimize(objective, np.array(x_opt), method='COBYLA', constraints=cons, bounds=bnds, options=options)
    #print('success: '+str(sol.success))
    #print('message: '+str(sol.message))
    x_opt = sol.x[:]
    w = x_opt[:m]
    b = x_opt[m]
    g = x_opt[m+1:m+n+1]
    #print('nfev= '+str(sol.nfev))
    #print('maxcv= '+str(sol.maxcv))
    print('w= '+str(w))
    print('b= '+str(b))
    if np.linalg.norm([w[i]-w_p[i] for i in range(0, m)]) < delta and sol.success:
        break
    nit += 1
    if sol.success:
        ub /= 2

#  restoring h
#  h[:n] - h1s, h[n:2*n] - h2s
print('constr= '+str(class_constr_inf_ineq(x_opt, w_p, ub)))
print('maxcv= '+ str(min(class_constr_inf_ineq(x_opt, w_p, ub))))


def obj_h(h):
    return np.dot(h, h)


def constr_h(h, w, g):
    ret = []
    for i in range(0, n):
        ret.append(h[i]*w[0]+h[n+i]*w[1]-g[i])
    return ret

h0 = np.array([0.1 for i in range(0, 2*n)])
con_h = {'type': 'eq', 'fun': constr_h, 'args': [w, g]}
cons_h = ([con_h])
solution_h = minimize(obj_h, h0, bounds=None, constraints=cons_h)
print(solution_h.success)
print(solution_h.message)
print(solution_h.nit)
h = solution_h.x

dataset_infected = []
print('attack norm= '+str(np.dot(h, h)/n))
print('objective value= '+str(sol.fun))
for i in range(0, n):
    tmp = []
    for j in range(0, m):
        tmp.append(dataset[i][j]+h[j*n+i])
    dataset_infected.append(tmp)


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