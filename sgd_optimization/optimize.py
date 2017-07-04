import random
import numpy as np
import datetime.datetime as dt
import sgd_optimization.obj_con_functions_v2 as of2


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
