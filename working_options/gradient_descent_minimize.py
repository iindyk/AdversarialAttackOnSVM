import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn import svm
from sklearn.metrics import accuracy_score

import utils.ofs.obj_con_functions_v1 as of1
from utils.datasets_parsers.random_dataset_generator import generate_random_dataset as grd

n = 150  # training set size
m = 2  # features
C = 1.0  # SVM regularization parameter
flip_size = 0  # random attack size
A = 0  # left end of interval for generating points
B = 100  # right end of interval for generating points
eps = 0.1*(B-A)  # upper bound for (norm of h)**2
maxit = 50  # maximum number of iterations
delta = 1e-2  # precision for break from iterations
options = {}  # solver options
learning_rate = 1e-5  # gradient update rate


dataset, labels, colors = grd(n=n, m=m, a=A, b=B, attack=flip_size, read=False, write=False, sep='linear')
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labels = svc.predict(dataset)
err_orig = 1 - accuracy_score(labels, predicted_labels)
print('err on orig is '+str(int(err_orig*100))+'%')
nit = 0
w = svc.coef_[0][:]
b = svc.intercept_
obj = 1e10
h = np.zeros(m*n)
while nit < maxit:
    print('iteration ' + str(nit) + '; start: ' + str(datetime.datetime.now().time()))
    h_p = h[:]
    obj_p = obj
    grad = of1.adv_obj_gradient(list(w)+[b]+[0.0 for i in range((m+2)*n)], dataset, labels)
    w = w - learning_rate*grad[:m]
    b = b - learning_rate*grad[m]
    con = {'type': 'ineq', 'fun': lambda x: n*eps - np.dot(x, x)}
    cons = [con]
    sol = minimize(lambda x: of1.class_obj_inf(w, b, x, dataset, labels, C), np.zeros(m*n), constraints=cons)
    if sol.success:
        h = sol.x[:]
    print('success: '+str(sol.success))
    print('message: '+str(sol.message))
    print('nfev= '+str(sol.nfev))
    print('w= '+str(w))
    print('b= '+str(b))
    print('attack_norm= '+str(100*np.dot(h, h)//(n*eps))+'%')
    obj = of1.adv_obj(list(w)+[b]+[0.0 for i in range((m+2)*n)], dataset, labels)
    nit += 1
    dataset_inf = np.array(dataset) + np.transpose(np.reshape(h, (m, n)))
    svc = svm.SVC(kernel='linear', C=C).fit(dataset_inf, labels)
    if (obj_p - obj < delta and np.dot(h, h) >= n*eps - delta) or not sol.success\
            or of1.coeff_diff(w, svc.coef_[0], b, svc.intercept_) > delta:
        break

dataset_inf = np.array(dataset) + np.transpose(np.reshape(h_p, (m, n)))
indices = range(n)
n_t = n
# define infected points for graph
inf_points = []
k = 0
for i in indices:
    if sum([h_p[j*n+k]**2 for j in range(m)]) > 0.9*eps:
        inf_points.append(dataset_inf[i])
    k += 1
svc1 = svm.SVC(kernel='linear', C=C)
svc1.fit(dataset_inf, labels)
predicted_labels_inf_svc = svc1.predict(dataset)
err_inf_svc = 1 - accuracy_score(labels, predicted_labels_inf_svc)
print('err on infected dataset by svc is '+str(int(100*err_inf_svc))+'%')
predicted_labels_inf_opt = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
err_inf_opt = 1 - accuracy_score(labels, predicted_labels_inf_opt)
print('err on infected dataset by opt is '+str(int(100*err_inf_opt))+'%')

# plots
if m == 2:
    plt.subplot(321)
    plt.title('original')
    plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
    plt.subplot(322)
    plt.title('infected')
    plt.scatter([float(i[0]) for i in dataset_inf], [float(i[1]) for i in dataset_inf], c=colors, cmap=plt.cm.coolwarm)
    plt.subplot(323)
    step = 1  # step size in the mesh

    x_min, x_max = int(A-2*(eps/m)**0.5), int(B+2*(eps/m)**0.5)
    y_min, y_max = int(A-2*(eps/m)**0.5), int(B+2*(eps/m)**0.5)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = np.sign([i[0]*w[0]+i[1]*w[1]+b for i in np.c_[xx.ravel(), yy.ravel()]])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset_inf], [float(i[1]) for i in dataset_inf], c=colors, cmap=plt.cm.coolwarm)
    plt.title('opt on inf data')
    plt.plot([i[0] for i in inf_points], [i[1] for i in inf_points], 'go', mfc='none')
    plt.subplot(324)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset_inf], [float(i[1]) for i in dataset_inf], c=colors, cmap=plt.cm.coolwarm)
    plt.plot([i[0] for i in inf_points], [i[1] for i in inf_points], 'go', mfc='none')
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
