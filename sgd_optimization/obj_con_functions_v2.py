import numpy as np


def adv_obj(w, b, dataset, labels):
    n = len(dataset)
    return sum([max(labels[i]*(np.dot(w, dataset[i]) + b), -1.0) for i in range(0, n)])/n


def adv_obj_gradient(w, b, dataset, labels):
    ret = []
    n, m = np.shape(dataset)
    for j in range(0, m):
        ret.append(sum([labels[i]*dataset[i][j]*(1.0 if labels[i]*(np.dot(w, dataset[i]) + b) > -1.0 else 0.0)
                        for i in range(0, n)]))  # with respect to w[j]
    ret.append(sum([labels[i]*(1.0 if labels[i]*(np.dot(w, dataset[i]) + b) > -1.0 else 0.0)
                    for i in range(0, n)]))  # with respect to b
    for i in range(0, (2+m)*n):
        ret.append(0.0)  # with respect to h, l, a
    return np.array(ret)


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
    # ret = np.append(ret,
    #               [0 if f >= 0 else 1 for f in class_constr_inf_ineq_nonconv(w, b, h, l, a, dataset, labels, eps, C)])
    ret = np.append(ret,
                    [0 if f >= 0 else -f for f in class_constr_inf_ineq_nonconv(w, b, h, l, a, dataset, labels, eps, C)])
    return ret


def class_constr_all_eq_trunc_matrix(w_prev, h_prev, l_prev, dataset, labels, eps, C):
    # x[:m]=2; x[m]=b; x[m+1:m+1+n*m]=h; x[m+1+n*m:m+1+n*(m+1)]=l; x[m+1+n*(m+1):m+1+n*(m+2)]=a
    n, m = np.shape(dataset)
    subset_num = 200
    subset_share = 4
    A = np.zeros((m + 3 + 3 * n + subset_num, m + (m + 2) * n + 1))
    b = np.zeros(m + 3 + 3 * n + subset_num)
    # w=\sum l_i * y_i *(x_i + h_i)
    for j in range(0, m):
        A[j][j] = -1.0
        for i in range(0, n):
            A[j][m+1+j*n+i] = l_prev[i]*labels[i]
            A[j][n*m+m+1+i] = labels[i]*dataset[i][j]
    # \sum l_i * y_i = 0
    for i in range(0, n):
        A[m][n*m+m+1+i] = labels[i]
    # l_i - l_i * a_i - l_i * y_i * (w * x_i + w * h_i + b) = 0
    for i in range(0, n):
        A[m+1+i][n*m+m+1+i] = 1.0 - labels[i]*np.dot(w_prev, dataset[i])
        A[m+1+i][m+1+(n+1)*m+i] = -l_prev[i]
        for j in range(0, m):
            A[m+1+i][m+1+n*j+i] = -l_prev[i]*labels[i]*w_prev[j]
        A[m+1+i][m] = -l_prev[i]*labels[i]
    # l_i * a_i = C * a_i
    for i in range(0, n):
        A[m+1+n+i][m+1+(n+1)*m+i] = C - l_prev[i]
    # y_i * (w * x_i + w * h_i + b) = 1 - a_i
    for i in range(0, n):
        for j in range(0, m):
            A[m+1+2*n+i][j] = labels[i]*dataset[i][j]
            A[m+1+2*n+i][m+1+j*n+i] = w_prev[j]*labels[i]
        A[m+1+2*n+i][m] = labels[i]
        A[m+1+2*n+i][m+1+n*(m+1)+i] = 1.0
        b[m+1+2*n+i] = 1.0
    # E||h||^2 = eps
    for i in range(0, n):
        for j in range(0, m):
            A[m+1+3*n][m+1+j*n+i] = h_prev[j*n+i]
    b[m+1+3*n] = eps*n
    # w[0] = 1
    A[m+2+3*n][0] = 1.0
    b[m+2+3*n] = 1.0
    # any* random subsample of size len(dataset)/subset_size of attack vectors has norm eps/subset_size
    indeces = np.random.randint(len(dataset), size=int(len(dataset) / subset_share))
    for k in range(subset_num):
        for i in indeces:
            for j in range(0, m):
                A[m+3+3*n+k][m+1+j*n+i] = h_prev[j*n+i]
        b[m+3+3*n+k] = eps*n / subset_share
    return A, b


