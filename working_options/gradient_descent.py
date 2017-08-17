from scipy.optimize import root, check_grad
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import sgd_optimization.obj_con_functions_v1 as of1
from datasets_parsers.random_dataset_generator import generate_random_dataset as grd


n = 50  # training set size
m = 2  # features
C = 1.0  # SVM regularization parameter
flip_size = 0  # random attack size
A = 0  # left end of interval for generating points
B = 100  # right end of interval for generating points
eps = 0.1*(B-A)  # upper bound for (norm of h)**2
maxit = 30  # maximum number of iterations
delta = 1e-2  # precision for break from iterations
options = {'diag': [50 for i in range(m*n)]+[1 for i in range(2*n+2+m+2*n)]}  # solver options
learning_rate = 1e-5  # gradient update rate


dataset, labels, colors = grd(n=n, m=m, a=A, b=B, attack=flip_size, read=False, write=False, sep='linear')
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labels = svc.predict(dataset)
err_orig = 1 - accuracy_score(labels, predicted_labels)
print('err on orig is '+str(int(err_orig*100))+'%')
nit = 0
w = svc.coef_[0]
b = svc.intercept_
z = np.zeros(m*n+2*n+2+m+2*n)
for i in range(n):
    if i in svc.support_:
        z[m*n+i] = svc.dual_coef_[0][list(svc.support_).index(i)] / labels[i]
    elif labels[i] * (np.dot(w, dataset[i]) + b) < 0:
        z[m * n + i] = C
    z[m*n+n+i] = max(0, 1-labels[i]*(np.dot(w, dataset[i])+b))
obj = 1e10
while nit < maxit:
    print('iteration ' + str(nit) + '; start: ' + str(datetime.datetime.now().time()))
    obj_p = obj
    grad = of1.adv_obj_gradient(list(w)+[b]+list(z), dataset, labels)
    if abs(of1.class_constr_nonconvex_all_as_eq(z, w, b, dataset, labels, C, eps)).sum() < delta:
        w = w - learning_rate*grad[:m]
        b = b - learning_rate*grad[m]
        print('making gradient update')
    sol = root(of1.class_constr_nonconvex_all_as_eq, z, args=(w, b, dataset, labels, C, eps), method='lm',
               jac=of1.class_constr_nonconvex_all_as_eq_jac, options=options)
    if sol.success:
        z = sol.x
    print('success: '+str(sol.success))
    print('message: '+str(sol.message))
    print('maxcv= '+str(max(abs(of1.class_constr_nonconvex_all_as_eq(z, w, b, dataset, labels, C, eps)))))
    print('nfev= '+str(sol.nfev))
    print('w= '+str(w))
    print('b= '+str(b))
    print('attack_norm= '+str(100*np.dot(z[:m*n], z[:m*n])//(n*eps))+'%')
    obj = of1.adv_obj(list(w)+[b]+list(z), dataset, labels)
    nit += 1
    if (obj_p - obj < delta and np.dot(z[:m*n], z[:m*n]) >= n*eps - delta) or not sol.success:
        break

h = z[:m*n]
dataset_infected = []
indices = range(n)
n_t = n
print('attack norm= '+str(np.dot(h, h)/n))
k = 0
for i in range(0, n):
    tmp = []
    if i in indices:
        for j in range(0, m):
            tmp.append(dataset[i][j]+h[j*n_t+k])
        k += 1
    else:
        tmp = dataset[i]
    dataset_infected.append(tmp)
# define infected points for graph
inf_points = []
k = 0
for i in indices:
    if sum([h[j*n+k]**2 for j in range(m)]) > 0.9*eps:
        inf_points.append(dataset_infected[i])
    k += 1
svc1 = svm.SVC(kernel='linear', C=C)
svc1.fit(dataset_infected, labels)
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
    plt.plot([i[0] for i in inf_points], [i[1] for i in inf_points], 'go', mfc='none')
    plt.subplot(324)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
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
