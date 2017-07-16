import random
import numpy as np
import datetime as dt
import sgd_optimization.obj_con_functions_v2 as of2
import sgd_optimization.obj_con_functions_v1 as of1
from scipy.optimize import minimize, lsq_linear
from sklearn import svm


def project_subspace(y, A, b, C, epsh, n_d, m_d, eps=np.finfo(float).eps):
    """ Project a vector onto the subspace defined by "dot(A,x) = b".
    """
    m, n = A.shape
    try:
        u, s, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError as e:
        print(e)
        return y / 100
    # Find the first singular value to drop below the cutoff.
    bad = (s < s[0] * eps)
    i = bad.searchsorted(1)
    if i < m:
        rcond = s[i]
    else:
        rcond = -1
    #x0 = np.linalg.lstsq(A, b, rcond=rcond)[0]
    lb = []
    ub = []
    # bounds for w:
    for j in range(m_d):
        lb.append(-1.0)
        ub.append(1.0)
    # bounds for b:
    lb.append(-np.inf)
    ub.append(np.inf)
    # bounds for h:
    for i in range(n_d):
        for j in range(m_d):
            lb.append(-np.sqrt(n_d * epsh))
            ub.append(np.sqrt(n_d * epsh))
    # bounds for l:
    for i in range(n_d):
        lb.append(0.0)
        ub.append(C)
    # bounds for a:
    for i in range(n_d):
        lb.append(0.0)
        ub.append(np.inf)
    sol = lsq_linear(A, b, bounds=(lb, ub))
    x0 = sol.x
    null_space = vh[i:]
    y_proj = x0 + (null_space * np.dot(null_space, y - x0)[:, np.newaxis]).sum(axis=0)
    return y_proj


def project_subspace_with_constr_minimize(y, w_prev, l_prev, dataset, labels, eps, C):
    # takes too long
    x0 = np.zeros_like(y)
    con1 = {'type': 'ineq', 'fun': of1.class_constr_inf_ineq_convex, 'args': [w_prev, dataset, labels, eps, C]}
    con2 = {'type': 'ineq', 'fun': of1.class_constr_inf_eq_convex, 'args': [w_prev, l_prev, dataset, labels, C]}
    con3 = {'type': 'ineq', 'fun': lambda x: -1 * of1.class_constr_inf_eq_convex(x, w_prev, l_prev, dataset, labels, C)}
    cons = [con1, con2, con3]
    sol = minimize(lambda x: sum([(x[i] - y[i]) ** 2 for i in range(0, len(x0))]), x0,
                   constraints=cons, method='COBYLA')
    if not sol.success:
        print('projection failed: ', sol.message)
    return sol.x


def projective_gradient_descent(dataset_full, labels_full, eps, C, batch_size=-1, maxit=100,
                                precision=1e-4, info=True, lrate=1e-2):
    nit = 0
    n, m = np.shape(dataset_full)
    # x = np.random.normal(1.0/n, 1.0 / (3.0 * n), m+1+n*(m+2))
    x = np.zeros(m + 1 + n * (m + 2))
    x[:m + 1] = np.random.normal(size=m + 1)
    # x[m + 1:m + m * n + 1] = np.random.normal((m * n))
    # x[m + 1:m + m * n + 1] = np.sqrt(n * eps) * x[m + 1:m + m * n + 1] / np.linalg.norm(x[m + 1:m + m * n + 1])
    x[m + 1:m + m * n + 1] = np.random.choice([0.5, -0.5], size=m*n)
    x[m + m * n + 1:] = np.random.normal(C / 2, C / 6, 2 * n)
    x_prev = np.zeros(m + 1 + n * (m + 2))
    if batch_size == -1: batch_size = len(dataset_full) / 10
    while nit < maxit and np.linalg.norm(x - x_prev) > precision:
        if info:
            print('Iteration ', nit, '; start at ', dt.datetime.now().time())
            print('w = ', x[:m], 'b = ', x[m])
            print('attack norm = ', np.dot(x[m + 1:m + 1 + m * n], x[m + 1:m + 1 + m * n] / n))
            print('objective value = ', of2.adv_obj_with_attack_norm(x[:m], x[m], x[m + 1:m + 1 + n * m],
                                                                     dataset_full, labels_full, eps))
        # indices = random.sample(range(0, len(dataset_full)), batch_size)
        # dataset = [dataset_full[i] for i in indices]
        # labels = [labels_full[i] for i in indices]
        x_prev = x[:]
        grad = of2.adv_obj_gradient_with_attack_norm(x[:m], x[m], x[m + 1:m + 1 + n * m], dataset_full, labels_full)
        A, b = of2.class_constr_all_eq_trunc_matrix(x[:m], x[m + 1:m + 1 + n * m], x[m + 1 + n * m:m + 1 + n * (m + 1)],
                                                    dataset_full, labels_full, eps, C)
        x = project_subspace(x - lrate * grad, A, b, C, epsh=eps, n_d=n, m_d=m)
        # x = project_subspace_with_constr_minimize(x - lrate*grad, x[:m], x[m+1+n*m:m+1+n*(m+1)],
        #                                          dataset_full, labels_full, eps, C)
        nit += 1
    return x[:m], x[m], x[m + 1:m + 1 + n * m]


def slsqp_optimization_with_gradient_convex(dataset_full, labels_full, eps, C, batch_size=-1, maxit=100,
                                            precision=1e-4, info=True):
    nit = 0
    n, m = np.shape(dataset_full)
    # x = np.random.normal(1.0/n, 1.0 / (3.0 * n), m+1+n*(m+2))
    x = np.zeros(m + 1 + n * (m + 2))
    x[:m + 1] = np.random.normal(m + 1)
    x[m + 1:m + m * n + 1] = np.random.normal(m * n)
    x[m + 1:m + m * n + 1] = np.sqrt(n * eps) * x[m + 1:m + m * n + 1] / np.linalg.norm(x[m + 1:m + m * n + 1])
    x[m + m * n + 1:] = np.random.normal(C / 2, C / 6, 2 * n)
    x_prev = np.zeros(m + 1 + n * (m + 2))
    if batch_size == -1: batch_size = len(dataset_full) / 10
    while nit < maxit and np.linalg.norm(x - x_prev) > precision:
        if info:
            print('Iteration ', nit, '; start at ', dt.datetime.now().time())
            print('w = ', x[:m], 'b = ', x[m])
            print('attack norm = ', np.dot(x[m + 1:m + 1 + m * n], x[m + 1:m + 1 + m * n] / n))
        x_prev = x[:]
        # todo
        nit += 1
    return x[:m], x[m], x[m + 1:m + 1 + n * m]


def slsqp_optimization_with_gradient_nonconvex(dataset_full, labels_full, eps, C, info=True):
    n, m = np.shape(dataset_full)
    x0 = np.zeros(m + 1 + n * (m + 2))
    svc = svm.SVC(kernel='linear', C=C).fit(dataset_full, labels_full)
    l_opt = []
    for i in range(0, n):
        if i in svc.support_:
            l_opt.append(svc.dual_coef_[0][list(svc.support_).index(i)] / labels_full[i])
        else:
            l_opt.append(0.0)
    x0[:m] = 1.0
    x0[m] = -100.0
    x0[m + 1 + m * n:m + 1 + n + m * n] = l_opt
    # x0[m + 1:m + m * n + 1] = np.random.normal(size=(m * n))
    # x0[m + 1:m + m * n + 1] = np.sqrt(n * eps) * x0[m + 1:m + m * n + 1] / np.linalg.norm(x0[m + 1:m + m * n + 1])
    # x0[m + m * n + 1:] = np.random.normal(C / 2, C / 6, 2 * n)
    con1 = {'type': 'ineq', 'fun': of1.class_constr_inf_ineq_nonconvex, 'jac': of1.class_constr_inf_ineq_nonconvex_jac,
            'args': [dataset_full, labels_full, eps]}
    con2 = {'type': 'eq', 'fun': of1.class_constr_inf_eq_nonconvex, 'jac': of1.class_constr_inf_eq_nonconvex_jac,
            'args': [dataset_full, labels_full, C]}
    cons = [con1, con2]
    bnds = []
    # bounds for w:
    for j in range(m):
        bnds.append((-1.0, 1.0))
    # bounds for b:
    bnds.append((None, None))
    # bounds for h:
    for i in range(n):
        for j in range(m):
            bnds.append((-np.sqrt(n * eps), np.sqrt(n * eps)))
    # bounds for l:
    for i in range(n):
        bnds.append((0.0, C))
    # bounds for a:
    for i in range(n):
        bnds.append((0.0, None))
    options = {'maxiter': 10000}
    sol = minimize(lambda x: -1.0 * of1.adv_obj(x, dataset_full, labels_full), x0, method='COBYLA',
                   jac=lambda x: -1.0 * of1.adv_obj_gradient(x, dataset_full, labels_full),
                   bounds=bnds, constraints=cons, options=options)
    x_opt = sol.x[:]
    if info:
        print('Iterations: ', sol.nit, '; success ', sol.success)
        print(sol.message)
        print('w = ', x_opt[:m], 'b = ', x_opt[m])
    # todo
    return x_opt[:m], x_opt[m], x_opt[m + 1:m + 1 + n * m]
