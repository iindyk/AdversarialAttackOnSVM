import numpy as np


def adv_obj(w, b, dataset, labels):
    n = len(dataset)
    return sum([max(labels[i]*(np.dot(w, dataset[i]) + b), -1.0) for i in range(0, n)])/n


def adv_obj_gradient(w, b, dataset, labels):
    ret = []
    n = len(dataset)
    m = len(dataset[0])
    for j in range(0, m):
        ret.append(sum([labels[i]*dataset[i][j]*(1.0 if labels[i]*(np.dot(w, dataset[i]) + b) > -1.0 else 0.0)
                        for i in range(0, n)]))  # with respect to w[j]
    ret.append(sum([labels[i]*(1.0 if labels[i]*(np.dot(w, dataset[i]) + b) > -1.0 else 0.0)
                    for i in range(0, n)]))  # with respect to b
    return ret


def class_constr_inf_eq_conv(w, b, h, l, a, w_prev, l_prev, dataset, labels, C):
    ret = []
    n = len(dataset)
    m = len(dataset[0])
    for j in range(0, m):
        ret.append(w[j] - sum([l_prev[i]*labels[i]*(dataset[i][j]+h[j*n+i]) for i in range(0, n)]))
    ret.append(sum([l[i]*labels[i] for i in range(0, n)]))
    for i in range(0, n):
        hi = [h[j * n + i] for j in range(0, m)]
        ret.append(l[i] - l_prev[i]*a[i] - l_prev[i]*labels[i]*(np.dot(w, dataset[i])+np.dot(w_prev, hi) + b))
        ret.append(l_prev[i]*a[i] - C*a[i])
    return np.array(ret)


def class_constr_inf_eq_nonconv(w, b, h, l, a, dataset, labels, C):
    ret = []
    n = len(dataset)
    m = len(dataset[0])
    for j in range(0, m):
        ret.append(w[j] - sum([l[i]*labels[i]*(dataset[i][j]+h[j*n+i]) for i in range(0, n)]))
    ret.append(sum([l[i]*labels[i] for i in range(0, n)]))
    for i in range(0, n):
        hi = [h[j * n + i] for j in range(0, m)]
        ret.append(l[i] - l[i]*a[i] - l[i]*labels[i]*(np.dot(w, dataset[i])+np.dot(w, hi) + b))
        ret.append(l[i]*a[i] - C*a[i])
    return np.array(ret)


def class_constr_inf_ineq_conv(w, b, h, l, a, w_prev, dataset, labels, eps, C):
    ret = []
    n = len(dataset)
    m = len(dataset[0])
    for i in range(0, n):
        ret.append(l[i])
        ret.append(C - l[i])
        ret.append(a[i])
        ret.append(labels[i]*(np.dot(w, dataset[i]) + np.dot(w_prev, [h[j * n + i] for j in range(0, m)])+b)-1+a[i])
    ret.append(eps*n - np.dot(h, h))
    #ret.append(1 - np.dot(w, w))
    #ret.append(1 - err_orig - adv_obj(x))
    return np.array(ret)


def class_constr_inf_ineq_nonconv(w, b, h, l, a, dataset, labels, eps, C):
    ret = []
    n = len(dataset)
    m = len(dataset[0])
    for i in range(0, n):
        ret.append(l[i])
        ret.append(C - l[i])
        ret.append(a[i])
        ret.append(labels[i]*(np.dot(w, dataset[i]) + np.dot(w, [h[j * n + i] for j in range(0, m)])+b)-1+a[i])
    ret.append(eps*n - np.dot(h, h))
    #ret.append(1 - np.dot(w, w))
    #ret.append(1 - err_orig - adv_obj(x))
    return np.array(ret)


def class_constr_all_eq(w, b, h, l, a, dataset, labels, eps, C):
    ret = np.array([])
    ret = np.append(ret, class_constr_inf_eq_nonconv(w, b, h, l, a, dataset, labels, C))
    #ret = np.append(ret,
    #                [0 if f >= 0 else 1 for f in class_constr_inf_ineq_nonconv(w, b, h, l, a, dataset, labels, eps, C)])
    ret = np.append(ret,
                    [0 if f >= 0 else -f for f in class_constr_inf_ineq_nonconv(w, b, h, l, a, dataset, labels, eps, C)])
    return ret


def class_constr_all_eq_trunc_matrix(w, b, h, l, a, dataset, labels, eps, C):
    n = len(dataset)
    m = len(dataset[0])
    A = np.zeros(m+(m+2)*n+1, m+(m+2)*n+1)
    b = np.zeros(m+(m+2)*n+1)