import datetime
import numpy as np
from scipy.optimize import minimize
import sgd_optimization.obj_con_functions_v1 as of1
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
flip_size = 0  # random attack size
A = 0
B = 100
eps = 1.0*(B-A)  # upper bound for (norm of h)**2
maxit = 20
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


l_opt = []
h_opt = []
for i in range(0, n):
    if i in svc.support_:
        #l_opt.append(1.0/n)
        l_opt.append(svc.dual_coef_[0][list(svc.support_).index(i)]/labels[i])
    else:
        l_opt.append(0.0)
x_opt = list(svc.coef_[0]) + list(svc.intercept_)+[np.sqrt(eps)/(m*n) for i in range(0, m*n)] + l_opt + [0.0 for i in range(0, n)]

options = {'maxiter': 10000}
nit = 0
while nit < maxit:
    print('iteration '+str(nit)+'; start: '+str(datetime.datetime.now().time()))
    w_p = x_opt[:m]
    l_p = x_opt[m*n+m+1:(m+1)*n+m+1]
    con1 = {'type': 'ineq', 'fun': of1.class_constr_inf_eq_convex, 'args': [w_p, l_p, dataset, labels, C]}
    con2 = {'type': 'ineq', 'fun': lambda x: -1*of1.class_constr_inf_eq_convex(x, w_p, l_p, dataset, labels, C)}
    con3 = {'type': 'ineq', 'fun': of1.class_constr_inf_ineq_convex_cobyla, 'args': [w_p, dataset, labels, eps, C]}
    cons = [con1, con2, con3]
    sol = minimize(of1.adv_obj, x_opt, args=(dataset, labels), constraints=cons, options=options, method='COBYLA')
    print('success: '+str(sol.success))
    print('message: '+str(sol.message))
    x_opt = sol.x[:]
    w_l, b_l, h_l, l_l, a_l = of1.decompose_x(x_opt, m, n)
    print('nfev= '+str(sol.nfev))
    #print('maxcv= '+str(sol.maxcv))
    print('w= '+str(w_l))
    print('b= '+str(b_l))
    x_p = list(w_p) + [b_l] + list(h_l) + list(l_p) + list(a_l)
    if of1.adv_obj(x_p, dataset, labels) <= sol.fun+delta \
            and max(of1.class_constr_inf_eq_convex(x_p, w_p, l_p, dataset, labels, C)) <= delta \
            and min(of1.class_constr_inf_eq_convex(x_p, w_p, l_p, dataset, labels, C)) >= -delta \
            and min(of1.class_constr_inf_ineq_convex_cobyla(x_p, w_p, dataset, labels, eps, C)) >= -delta \
            and sol.success and np.dot(h_l, h_l) / n > eps - 0.1:
        break
    nit += 1

dataset_infected = []
print('attack norm= '+str(np.dot(h_l, h_l)/n))
print('objective value= '+str(sol.fun))
for i in range(0, n):
    tmp = []
    for j in range(0, m):
        tmp.append(dataset[i][j]+h_l[j*n+i])
    dataset_infected.append(tmp)

svc1 = svm.SVC(kernel='linear', C=C)
svc1.fit(dataset_infected, labels)
predicted_labels_inf_svc = svc1.predict(dataset)
err_inf_svc = 1 - accuracy_score(labels, predicted_labels_inf_svc)
print('err on infected dataset by svc is '+str(err_inf_svc))
predicted_labels_inf_opt = np.sign([np.dot(dataset[i], w_l)+b_l for i in range(0, n)])
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
Z = np.sign([i[0]*w_l[0]+i[1]*w_l[1]+b_l for i in np.c_[xx.ravel(), yy.ravel()]])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
plt.title('opt on inf data')
plt.subplot(324)
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
plt.title('inf svc on inf data')

plt.subplot(326)
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.title('inf svc on orig data')

plt.subplot(325)
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.title('orig svc on orig data')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()