import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn import svm
from sklearn.metrics import accuracy_score
import sgd_optimization.obj_con_functions_v1 as of1
from datasets_parsers.random_dataset_generator import generate_random_dataset as grd
from datasets_parsers.get_truncated_dataset import truncate_by_dist as trunc

n = 200  # training set size
m = 2  # features
C = 1.0  # SVM regularization parameter
flip_size = 0  # random attack size
A = 0  # left end of interval for generating points
B = 100  # right end of interval for generating points
eps = 0.1*(B-A)  # upper bound for (norm of h)**2
maxit = 20  # maximum number of iterations
delta = 1e-2  # precision for break from iterations
options = {'maxiter': 10000}  # solver options
maxdist = 0.1*(B-A)  # maximum allowed distance for dataset truncating


dataset, labels, colors = grd(n=n, m=m, a=A, b=B, attack=flip_size, read=False, write=False, sep='linear')
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labels = svc.predict(dataset)
err_orig = 1 - accuracy_score(labels, predicted_labels)
print('err on orig is '+str(int(err_orig*100))+'%')
dataset_trunc, labels_trunc, colors_trunc, indices, x_opt, eps_t = trunc(dataset, labels, colors, maxdist, C, eps)
n_t = len(dataset_trunc)
print('number of closest chosen points is '+str(n_t))
w = x_opt[:m]
l = x_opt[m+1+m*n_t:m+1+(m+1)*n_t]
nit = 0
while nit < maxit:
    print('iteration '+str(nit)+'; start: '+str(datetime.datetime.now().time()))
    con1 = {'type': 'ineq', 'fun': of1.class_constr_inf_eq_convex,
            'args': [w, l, dataset_trunc, labels_trunc, C]}
    con2 = {'type': 'ineq',
            'fun': lambda x: -1*of1.class_constr_inf_eq_convex(x, w, l, dataset_trunc, labels_trunc, C)}
    con3 = {'type': 'ineq', 'fun': of1.class_constr_inf_ineq_convex_cobyla,
            'args': [w, dataset_trunc, labels_trunc, eps_t, C]}
    cons = [con1, con2, con3]
    sol = minimize(of1.adv_obj, x_opt, args=(dataset_trunc, labels_trunc), constraints=cons, options=options, method='COBYLA')
    print('success: '+str(sol.success))
    print('message: '+str(sol.message))
    x_opt = sol.x[:]
    w, b, h, l, a = of1.decompose_x(x_opt, m, n_t)
    print('nfev= '+str(sol.nfev))
    print('w= '+str(w))
    print('b= '+str(b))
    print('attack_norm= '+str(100*np.dot(h, h)//(n_t*eps_t))+'%')
    if of1.adv_obj(x_opt, dataset_trunc, labels_trunc) <= sol.fun+delta \
            and max(of1.class_constr_inf_eq_convex(x_opt, w, l, dataset_trunc, labels_trunc, C)) <= delta \
            and min(of1.class_constr_inf_eq_convex(x_opt, w, l, dataset_trunc, labels_trunc, C)) >= -delta \
            and min(of1.class_constr_inf_ineq_convex_cobyla(x_opt, w, dataset_trunc, labels_trunc, eps_t, C)) >= -delta \
            and sol.success and np.dot(h, h) / n_t >= eps_t - delta:
        break
    nit += 1

dataset_infected = []
print('attack norm= '+str(np.dot(h, h)/n))
print('objective value= '+str(sol.fun))
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
    if sum([h[j*n_t+k]**2 for j in range(m)]) > 0.9*eps:
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