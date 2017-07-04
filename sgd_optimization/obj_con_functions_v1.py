import numpy as np


def decompose_x(x, m, n):
    return x[:m], x[m], x[m+1:m*n+m+1], \
           x[m*n+m+1:(m+1)*n+m+1], x[(m+1)*n+m+1:(m+2)*n+m+1]   # w, b, h, l, a


def approx_fun(x):
    return max(x, -1.0)
    #return min(x, 0.0)
    #return 1.0 if x > 0 else 0.0
    #return x
    #return -(1.0 if x <-1.0 else 2.0-np.exp(1+x))
    #return -(1.0 if x < -1.0 else -x**2 - 2*x)
    #return -0.5+2*sum([np.sin(np.pi*(2*k+1)*x/100)/(np.pi*(2*k+1)) for k in range(0, 100)])


def adv_obj(x, dataset, labels):
    n = len(dataset)
    m = len(dataset[0])
    av = 0.0
    for i in range(0, n):
        av += approx_fun(labels[i]*(np.dot(x[:m], dataset[i]) + x[m]))
    return av/n


def class_constr_inf_eq(x, w_prev, l_prev, dataset, labels, C):
    ret = []
    n = len(dataset)
    m = len(dataset[0])
    w, b, h, l, a = decompose_x(x, m, n)
    for j in range(0, m):
        ret.append(w[j] - sum([l_prev[i]*labels[i]*(dataset[i][j]+h[j*n+i]) for i in range(0, n)]))
    ret.append(sum([l[i]*labels[i] for i in range(0, n)]))
    for i in range(0, n):
        hi = [h[j * n + i] for j in range(0, m)]
        ret.append(l[i] - l_prev[i]*a[i] - l_prev[i]*labels[i]*(np.dot(w, dataset[i])+np.dot(w_prev, hi) + b))
        ret.append(l_prev[i]*a[i] - C*a[i])
    return np.array(ret)


def class_constr_inf_ineq(x, w_prev, dataset, labels, eps, C):
    ret = []
    n = len(dataset)
    m = len(dataset[0])
    w, b, h, l, a = decompose_x(x, m, n)
    for i in range(0, n):
        ret.append(l[i])
        ret.append(C - l[i])
        ret.append(a[i])
        ret.append(labels[i]*(np.dot(w, dataset[i]) + np.dot(w_prev, [h[j * n + i] for j in range(0, m)])+b)-1+a[i])
    ret.append(eps*n - np.dot(h, h))
    #ret.append(1 - np.dot(w, w))
    #ret.append(1 - err_orig - adv_obj(x))
    return np.array(ret)