import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score


def decompose_x(x, m, n):
    return x[:m], x[m], x[m + 1:m * n + m + 1], \
           x[m * n + m + 1:(m + 1) * n + m + 1], x[(m + 1) * n + m + 1:(m + 2) * n + m + 1]  # w, b, h, l, a


def approx_fun(x):
    return max(x, -1.0)
    # return min(x, 0.0)
    # return 1.0 if x > 0 else 0.0
    # return x
    # return -(1.0 if x <-1.0 else 2.0-np.exp(1+x))
    # return -(1.0 if x < -1.0 else -x**2 - 2*x)
    # return -0.5+2*sum([np.sin(np.pi*(2*k+1)*x/100)/(np.pi*(2*k+1)) for k in range(0, 100)])


def coeff_diff(w1, w2, b1, b2):
    w1_n = w1 / np.sqrt(np.dot(w1, w1) + b1 ** 2)
    b1_n = b1 / np.sqrt(np.dot(w1, w1) + b1 ** 2)
    w2_n = w2 / np.sqrt(np.dot(w2, w2) + b2 ** 2)
    b2_n = b2 / np.sqrt(np.dot(w2, w2) + b2 ** 2)
    return np.sqrt(np.dot(w1_n - w2_n, w1_n - w2_n) + (b1_n - b2_n) ** 2)


def class_obj_inf(w, b, h, dataset, labels, C):
    av = 0.0
    n, m = np.shape(dataset)
    dataset_inf = np.array(dataset) + np.transpose(np.reshape(h, (m, n)))
    for i in range(n):
        av += max(0, 1 - labels[i]*(np.dot(w, dataset_inf[i])+b))
    return C*av + np.dot(w, w)/2


def adv_obj(x, dataset, labels):
    n = len(dataset)
    m = len(dataset[0])
    av = 0.0
    for i in range(0, n):
        av += approx_fun(labels[i] * (np.dot(x[:m], dataset[i]) + x[m]))
    return av / n


def adv_obj_gradient(x, dataset, labels):
    ret = []
    n, m = np.shape(dataset)
    for j in range(0, m):
        ret.append(
            sum([labels[i] * dataset[i][j] * (1.0 if labels[i] * (np.dot(x[:m], dataset[i]) + x[m]) > -1.0 else 0.0)
                 for i in range(0, n)]))  # with respect to w[j]
    ret.append(sum([labels[i] * (1.0 if labels[i] * (np.dot(x[:m], dataset[i]) + x[m]) > -1.0 else 0.0)
                    for i in range(0, n)]))  # with respect to b
    for i in range(0, (2 + m) * n):
        ret.append(0.0)  # with respect to h, l, a
    return np.array(ret)


def class_constr_inf_eq_convex(x, w_prev, l_prev, dataset, labels, C):
    ret = []
    n, m = np.shape(dataset)
    w, b, h, l, a = decompose_x(x, m, n)
    for j in range(0, m):
        ret.append(w[j] - sum([l_prev[i] * labels[i] * (dataset[i][j] + h[j * n + i]) for i in range(0, n)]))
    ret.append(sum([l[i] * labels[i] for i in range(0, n)]))
    for i in range(0, n):
        hi = [h[j * n + i] for j in range(0, m)]
        ret.append(l[i] - l_prev[i] * a[i] - l_prev[i] * labels[i] * (np.dot(w, dataset[i]) + np.dot(w_prev, hi) + b))
        ret.append(l_prev[i] * a[i] - C * a[i])
    return np.array(ret)


def class_constr_inf_eq_convex_jac(x, w_prev, l_prev, dataset, labels, C):
    n, m = np.shape(dataset)
    ret = np.zeros((m + 1 + 2 * n, m + 1 + (m + 2) * n))
    w, b, h, l, a = decompose_x(x, m, n)
    # d(con_eq[:m])/dx
    for i in range(m):
        # with respect to w
        ret[i, i] = 1.0
        # with respect to b = 0
        # with respect to h
        for j in range(n):
            ret[i, m + 1 + j + i * n] = -l[j] * labels[j]
            # with respect to l = 0
            # with respect to a = 0
    # d(con_eq[m])/dx
    for j in range(n):
        # with respect to l, everything else is 0
        ret[m, m + 1 + n * m + j] = labels[j]
    # d(con_eq[m+1:m+n+1])/dx
    for i in range(n):
        # with respect to w
        for j in range(m):
            ret[m + 1 + i, j] = -l_prev[i] * labels[i] * dataset[i][j]
        # with respect to b
        ret[m + 1 + i, m] = -l_prev[i] * labels[i]
        # with respect to h
        for j in range(m):
            ret[m + 1 + i][m + 1 + j * n + i] = -l_prev[i] * labels[i] * w_prev[j]
        # with respect to l
        ret[m + 1 + i][m + 1 + n * m + i] = 1.0
        # with respect to a
        ret[m + 1 + i][m + 1 + (m + 1) * n + i] = -l_prev[i]
    # d(con_eq[m+n+1:])/dx
    for i in range(n):
        # with respect to a, everything else is 0
        ret[m + 1 + n + i][m + 1 + m * n + n + i] = l_prev[i] - C
    return ret


def class_constr_inf_ineq_convex(x, w_prev, dataset, labels, eps, C):
    ret = []
    n, m = np.shape(dataset)
    w, b, h, l, a = decompose_x(x, m, n)
    for i in range(0, n):
        ret.append(
            labels[i] * (np.dot(w, dataset[i]) + np.dot(w_prev, [h[j * n + i] for j in range(0, m)]) + b) - 1 + a[i])
    ret.append(eps * n - np.dot(h, h))
    return np.array(ret)


def class_constr_inf_ineq_convex_cobyla(x, w_prev, dataset, labels, eps, C):
    ret = []
    n, m = np.shape(dataset)
    w, b, h, l, a = decompose_x(x, m, n)
    for i in range(0, n):
        ret.append(l[i])  # for cobyla only
        ret.append(C - l[i])  # for cobyla only
        ret.append(a[i])  # for cobyla only
        ret.append(
            labels[i] * (np.dot(w, dataset[i]) + np.dot(w_prev, [h[j * n + i] for j in range(0, m)]) + b) - 1 + a[i])
    ret.append(eps * n - np.dot(h, h))
    return np.array(ret)


def class_constr_inf_ineq_convex_jac(x, w_prev, dataset, labels, eps, C):
    n, m = np.shape(dataset)
    ret = np.zeros((n + 1, m + 1 + (m + 2) * n))
    w, b, h, l, a = decompose_x(x, m, n)
    # d(cons_ineq[:n])/dx
    for i in range(n):
        ret[0, 0] = 0
    # todo: finish
    return np.array(ret)


def class_constr_inf_eq_nonconvex(x, dataset, labels, C):
    ret = []
    n, m = np.shape(dataset)
    w, b, h, l, a = decompose_x(x, m, n)
    for j in range(0, m):
        ret.append(w[j] - sum([l[i] * labels[i] * (dataset[i][j] + h[j * n + i]) for i in range(0, n)]))
    ret.append(sum([l[i] * labels[i] for i in range(0, n)]))
    for i in range(0, n):
        hi = [h[j * n + i] for j in range(0, m)]
        ret.append(l[i] - l[i] * a[i] - l[i] * labels[i] * (np.dot(w, dataset[i]) + np.dot(w, hi) + b))
        ret.append(l[i] * a[i] - C * a[i])
    return np.array(ret)


def class_constr_inf_eq_nonconvex_jac(x, dataset, labels, C):
    n, m = np.shape(dataset)
    ret = np.zeros((m + 1 + 2 * n, m + 1 + (m + 2) * n))
    w, b, h, l, a = decompose_x(x, m, n)
    # d(con_eq[:m])/dx
    for i in range(m):
        # with respect to w
        ret[i, i] = 1.0
        # with respect to b = 0
        # with respect to h
        for j in range(n):
            ret[i, m + 1 + j + i * n] = -l[j] * labels[j]
        # with respect to l
        for j in range(n):
            ret[i, m + 1 + n * m + j] = -labels[j] * (dataset[j][i] + h[i * n + j])
            # with respect to a = 0
    # d(con_eq[m])/dx
    for j in range(n):
        # with respect to l, everything else is 0
        ret[m, m + 1 + n * m + j] = labels[j]
    # d(con_eq[m+1:m+n+1])/dx
    for i in range(n):
        # with respect to w
        for j in range(m):
            ret[m + 1 + i, j] = -l[i] * labels[i] * (dataset[i][j] + h[j * n + i])
        # with respect to b
        ret[m + 1 + i, m] = -l[i] * labels[i]
        # with respect to h
        for j in range(m):
            ret[m + 1 + i][m + 1 + j * n + i] = -l[i] * labels[i] * w[j]
        # with respect to l
        ret[m + 1 + i][m + 1 + n * m + i] = 1.0 - a[i] - labels[i] * (
            np.dot(w, dataset[i]) + sum([w[j] * h[j * n + i] for j in range(m)]) + b)
        # with respect to a
        ret[m + 1 + i][m + 1 + (m + 1) * n + i] = -l[i]
    # d(con_eq[m+n+1:])/dx
    for i in range(n):
        # with respect to l
        ret[m + 1 + n + i][m + 1 + m * n + i] = a[i]
        # with respect to a, everything else is 0
        ret[m + 1 + n + i][m + 1 + m * n + n + i] = l[i] - C
    return ret


def class_constr_inf_ineq_nonconvex(x, dataset, labels, eps):
    ret = []
    n, m = np.shape(dataset)
    w, b, h, l, a = decompose_x(x, m, n)
    for i in range(0, n):
        ret.append(labels[i] * (np.dot(w, dataset[i]) + np.dot(w, [h[j * n + i] for j in range(m)]) + b) - 1 + a[i])
    ret.append(eps * n - np.dot(h, h))
    return np.array(ret)


def class_constr_inf_ineq_nonconvex_jac(x, dataset, labels, eps):
    n, m = np.shape(dataset)
    ret = np.zeros((n + 1, m + 1 + (m + 2) * n))
    w, b, h, l, a = decompose_x(x, m, n)
    # d(cons_ineq[:n])/dx
    for i in range(n):
        # with respect to w
        for j in range(m):
            ret[i, j] = labels[i] * (dataset[i][j] + h[j * n + i])
        # with respect to b
        ret[i, m] = labels[i]
        # with respect to h
        for j in range(m):
            ret[i, m + 1 + j * n + i] = labels[i] * w[j]
        # with respect to l = 0
        # with respect to a
        ret[i, m + 1 + (m + 1) * n + i] = 1.0
    # d(cons_ineq[n])/dx
    # with respect to h, everything else is 0
    for i in range(n):
        for j in range(m):
            ret[n, m + 1 + j * n + i] = -2.0 * h[j * n + i]
    return ret


def ineq_to_eq(x):
    #ret = min(x, 0)
    ret = np.exp(-x**(-1)) if x > 0 else 0.0
    return ret


def ineq_to_eq_jac(x):
    #ret = 1.0 if x > 0 else 0
    ret = np.exp(-x**(-1))*(x**(-2)) if x > 0 else 0.0
    return ret


def class_constr_nonconvex_all_as_eq(z, w, b, dataset, labels, C, eps):
    n, m = np.shape(dataset)
    x = list(w) + [b] + list(z[:(m + 2) * n])
    w, b, h, l, a = decompose_x(x, m, n)
    ret = class_constr_inf_eq_nonconvex(x, dataset, labels, C)
    for cons in class_constr_inf_ineq_nonconvex(x, dataset, labels, eps):
        ret = np.append(ret, ineq_to_eq(cons))
    for i in range(n):
        ret = np.append(ret, ineq_to_eq(l[i]))
        ret = np.append(ret, ineq_to_eq(C - l[i]))
        ret = np.append(ret, ineq_to_eq(a[i]))
    return ret


def class_constr_nonconvex_all_as_eq_jac(z, w, b, dataset, labels, C, eps):
    n, m = np.shape(dataset)
    x = list(w) + [b] + list(z[:(m + 2) * n])
    ret = np.zeros((m + 1 + 2 * n + n + 1 + 3 * n, (m + 2) * n+2+m+2*n))
    w, b, h, l, a = decompose_x(x, m, n)
    cons_ineq = class_constr_inf_ineq_nonconvex(x, dataset, labels, eps)
    # d(con_eq[:m])/dz
    for i in range(m):
        # with respect to h
        for j in range(n):
            ret[i, j + i * n] = -l[j] * labels[j]
        # with respect to l
        for j in range(n):
            ret[i, n * m + j] = -labels[j] * (dataset[j][i] + h[i * n + j])
            # with respect to a = 0
    # d(con_eq[m])/dz
    for j in range(n):
        # with respect to l, everything else is 0
        ret[m, n * m + j] = labels[j]
    # d(con_eq[m+1:m+n+1])/dz
    for i in range(n):
        # with respect to h
        for j in range(m):
            ret[m + 1 + i, j * n + i] = -l[i] * labels[i] * w[j]
        # with respect to l
        ret[m + 1 + i, n * m + i] = 1.0 - a[i] - labels[i] * \
                                                 (np.dot(w, dataset[i]) + sum(
                                                     [w[j] * h[j * n + i] for j in range(m)]) + b)
        # with respect to a
        ret[m + 1 + i, (m + 1) * n + i] = -l[i]
    # d(con_eq[m+n+1:])/dz
    for i in range(n):
        # with respect to l
        ret[m + 1 + n + i, m * n + i] = a[i]
        # with respect to a, everything else is 0
        ret[m + 1 + n + i, m * n + n + i] = l[i] - C
    # d(cons_ineq[:n])/dz
    for i in range(n):
        # with respect to h
        for j in range(m):
            ret[m + 1 + 2 * n + i, j * n + i] = labels[i] * w[j] * ineq_to_eq_jac(cons_ineq[i])
        # with respect to l = 0
        # with respect to a
        ret[m + 1 + 2 * n + i, (m + 1) * n + i] = ineq_to_eq_jac(cons_ineq[i])
    # d(cons_ineq[n])/dz
    # with respect to h, everything else is 0
    for i in range(n):
        for j in range(m):
            ret[m + 1 + 2 * n + n, j * n + i] = -2.0 * h[j * n + i] * ineq_to_eq_jac(cons_ineq[n])
    # d(bounds)/dz
    for i in range(n):
        # l[i]>0
        ret[m + 1 + 2 * n + n + 1 + i, m * n + i] = ineq_to_eq_jac(l[i])
        # C-l[i]>0
        ret[m + 1 + 2 * n + 2 * n + 1 + i, m * n + i] = ineq_to_eq_jac(C - l[i]) * (-1)
        # a[i]>0
        ret[m + 1 + 2 * n + 3 * n + 1 + i, m * n + i] = ineq_to_eq_jac(a[i])
    return ret


def get_initial_data(dataset, labels, C, eps, svc_orig):
    n, m = np.shape(dataset)
    predicted_labels = svc_orig.predict(dataset)
    err_orig = 1 - accuracy_score(labels, predicted_labels)
    # initial x respects to zero attack and classifier parameters
    x0 = np.zeros(m + 1 + n * (m + 2))
    x0[:m] = svc_orig.coef_[0]
    x0[m] = svc_orig.intercept_
    for i in range(n):
        if i in svc_orig.support_:
            x0[m * n + m+1+i] = svc_orig.dual_coef_[0][list(svc_orig.support_).index(i)] / labels[i]
        elif labels[i] * (np.dot(x0[:m], dataset[i]) + x0[m]) < 0:
            x0[m * n + m + 1 + i] = C
        x0[m * n + n + i+m+1] = max((0, 1 - labels[i] * (np.dot(x0[:m], dataset[i]) + x0[m])))
    # define bounds for variables:
    x_L = []
    x_U = []
    # bounds for w:
    for j in range(m):
        x_L.append(-1.0)
        x_U.append(1.0)
    # bounds for b:
    x_L.append(-1e10)
    x_U.append(1e10)
    # bounds for h:
    for i in range(n):
        for j in range(m):
            x_L.append(-np.sqrt(n * eps))
            x_U.append(np.sqrt(n * eps))
    # bounds for l:
    for i in range(n):
        x_L.append(0.0)
        x_U.append(C)
    # bounds for a:
    for i in range(n):
        x_L.append(0.0)
        x_U.append(1e10)
    # define bounds for constraints:
    g_L = np.zeros((3*n+m+2))
    g_U = np.zeros((3*n+m+2))
    g_U[2*n+m+1:] = 1e10
    return x0, np.array(x_L), np.array(x_U), g_L, g_U, err_orig
