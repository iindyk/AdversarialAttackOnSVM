import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

import sgd_optimization.optimize as opt
from datasets_parsers.random_dataset_generator import generate_random_dataset as grd

eps = 100.0
n = 200
m = 2
C = 1.0/n
A = 0
B = 100
maxit = 200
lrate = 1e-3
dataset, labels, colors = grd(n=n, m=m, a=A, b=B, attack=0, read=False, write=False, sep='linear')
w, b, h = opt.projective_gradient_descent(dataset, labels, eps, C, lrate=lrate)

svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labels = svc.predict(dataset)
err_orig = 1 - accuracy_score(labels, predicted_labels)
print('err on orig is '+str(err_orig))

dataset_infected = []
print('attack norm= '+str(np.dot(h, h)/n))
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
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors, cmap=plt.cm.coolwarm)
plt.title('opt on inf data')
plt.subplot(324)
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.xlim(0, 100)
plt.ylim(0, 100)
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
