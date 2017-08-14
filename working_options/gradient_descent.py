from scipy.optimize import fsolve, root
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import sgd_optimization.obj_con_functions_v1 as of1
from datasets_parsers.random_dataset_generator import generate_random_dataset as grd


n = 5  # training set size
m = 2  # features
C = 1.0  # SVM regularization parameter
flip_size = 0  # random attack size
A = 0  # left end of interval for generating points
B = 100  # right end of interval for generating points
eps = 0.1*(B-A)  # upper bound for (norm of h)**2
maxit = 30  # maximum number of iterations
delta = 1e-2  # precision for break from iterations
options = {'maxiter': 10000}  # solver options


dataset, labels, colors = grd(n=n, m=m, a=A, b=B, attack=flip_size, read=False, write=False, sep='linear')
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
predicted_labels = svc.predict(dataset)
err_orig = 1 - accuracy_score(labels, predicted_labels)
print('err on orig is '+str(int(err_orig*100))+'%')
nit = 0
x0 = np.zeros(m*n+2*n)
w = [1, 1]
b = -100


def callback(x, f):
    print('hi, i am here')
    print('x = '+str(x))
ans = root(of1.class_constr_nonconvex_all_as_eq, x0, args=(w, b, dataset, labels, C, eps), method='broyden1',
           callback=callback)
print(ans.x)
print(ans.success)
print(ans.message)
# while nit < maxit:
