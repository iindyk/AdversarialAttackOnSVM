import random
import numpy as np
import datetime as dt
import sgd_optimization.obj_con_functions_v2 as of2


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


def sgd_adv_class_sbs(dataset_full, labels_full, eps, C, batch_size=-1, maxit=100, precision=1e-4, info=True, lrate=1e-2):
    nit = 0
    n, m = np.shape(dataset_full)
    x = np.random.normal(1.0/n, 1.0 / (3.0 * n), m+1+n*(m+2))
    x_prev = np.zeros(m+1+n*(m+2))
    if batch_size == -1: batch_size = len(dataset_full) / 10
    while nit < maxit and np.linalg.norm(x-x_prev) > precision:
        if info:
            print('Iteration ', nit, '; start at ', dt.datetime.now().time())
            print('w = ', x[:m], 'b = ', x[m])
            print('attack norm = ', np.dot(x[m+1:m+1+m*n], x[m+1:m+1+m*n]))
        # indices = random.sample(range(0, len(dataset_full)), batch_size)
        # dataset = [dataset_full[i] for i in indices]
        # labels = [labels_full[i] for i in indices]
        x_prev = x[:]
        grad = of2.adv_obj_gradient(x[:m], x[m], dataset_full, labels_full)
        A, b = of2.class_constr_all_eq_trunc_matrix(x[:m], x[m+1:m+1+n*m], x[m+1+n*m:m+1+n*(m+1)],
                                                    dataset_full, labels_full, eps, C)
        x = project_subspace(x - lrate*grad, A, b)
        nit += 1
    return x[:m], x[m], x[m+1:m+1+n*m]
