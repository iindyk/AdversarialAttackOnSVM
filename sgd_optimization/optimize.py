import random
import numpy as np
import datetime.datetime as dt
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


def sgd_adv_class_sbs(dataset_full, labels_full, eps, batch_size=-1, maxit=100, precision=1e-4, info=True, lrate=1e-2):
    nit = 0
    n = len(dataset_full)
    m = len(dataset_full[0])
    w = np.random.normal(m)
    b = np.random.normal(1)
    h = np.random.normal(0.0, eps / (3.0 * m), m * n)
    if batch_size == -1: batch_size = len(dataset_full) / 10
    while nit < maxit:
        if info: print('Iteration ', nit, '; start at ', dt.now().time())
        indices = random.sample(range(0, len(dataset_full)), batch_size)
        dataset = [dataset_full[i] for i in indices]
        labels = [labels_full[i] for i in indices]
        # solve adv
        grad = of2.adv_obj_gradient(w, b, dataset, labels)
        w = w - lrate*grad[:m]
        b = b - lrate*grad[m]
        # solve class
        nit += 1
    return w, b, h
