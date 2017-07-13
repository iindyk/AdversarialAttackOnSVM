import random
import numpy as np
import datetime as dt
import sgd_optimization.obj_con_functions_v2 as of2
import sgd_optimization.obj_con_functions_v1 as of1
from scipy.optimize import minimize


def project_subspace(y, A, b, eps=np.finfo(float).eps):
    """ Project a vector onto the subspace defined by "dot(A,x) = b".
    """
    m, n = A.shape
    u, s, vh = np.linalg.svd(A)
    # Find the first singular value to drop below the cutoff.
    bad = (s < s[0] * eps)
    i = bad.searchsorted(1)
    if i < m:
        rcond = s[i]
    else:
        rcond = -1
    x0 = np.linalg.lstsq(A, b, rcond=rcond)[0]
    null_space = vh[i:]
    y_proj = x0 + (null_space * np.dot(null_space, y-x0)[:, np.newaxis]).sum(axis=0)
    return y_proj


def project_subspace_with_constr_minimize(y, w_prev, l_prev, dataset, labels, eps, C):
    x0 = np.zeros_like(y)
    con1 = {'type': 'ineq', 'fun': of1.class_constr_inf_ineq, 'args': [w_prev, dataset, labels, eps, C]}
    con2 = {'type': 'eq', 'fun': of1.class_constr_inf_eq, 'args': [w_prev, l_prev, dataset, labels, C]}
    cons = [con1, con2]
    sol = minimize(lambda x: sum([(x[i]-y[i])**2 for i in range(len(x0))]), x0, constraints=cons)
    if not sol.success:
        print('projection failed: ', sol.message)
    return sol.x


def projective_gradient_descent(dataset_full, labels_full, eps, C, batch_size=-1, maxit=100, precision=1e-4, info=True, lrate=1e-2):
    nit = 0
    n, m = np.shape(dataset_full)
    # x = np.random.normal(1.0/n, 1.0 / (3.0 * n), m+1+n*(m+2))
    x = np.zeros(m+1+n*(m+2))
    x[:m+1] = np.random.normal(m+1)
    x[m+1:m+m*n+1] = np.random.normal(m*n)
    x[m+1:m+m*n+1] = np.sqrt(n*eps)*x[m+1:m+m*n+1]/np.linalg.norm(x[m+1:m+m*n+1])
    x[m+m*n+1:] = np.random.normal(C/2, C/6, 2*n)
    x_prev = np.zeros(m+1+n*(m+2))
    if batch_size == -1: batch_size = len(dataset_full) / 10
    while nit < maxit and np.linalg.norm(x-x_prev) > precision:
        if info:
            print('Iteration ', nit, '; start at ', dt.datetime.now().time())
            print('w = ', x[:m], 'b = ', x[m])
            print('attack norm = ', np.dot(x[m+1:m+1+m*n], x[m+1:m+1+m*n]/n))
        # indices = random.sample(range(0, len(dataset_full)), batch_size)
        # dataset = [dataset_full[i] for i in indices]
        # labels = [labels_full[i] for i in indices]
        x_prev = x[:]
        grad = of2.adv_obj_gradient(x[:m], x[m], x[m+1:m+n*m+1], dataset_full, labels_full, eps)
        # A, b = of2.class_constr_all_eq_trunc_matrix(x[:m], x[m+1:m+1+n*m], x[m+1+n*m:m+1+n*(m+1)],
        #                                             dataset_full, labels_full, eps, C)
        # x = project_subspace(x - lrate*grad, A, b)
        x = project_subspace_with_constr_minimize(x - lrate*grad, x[:m], x[m+1+n*m:m+1+(n+1)*m],
                                                  dataset_full, labels_full, eps, C)
        nit += 1
    return x[:m], x[m], x[m+1:m+1+n*m]
