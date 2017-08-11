import numpy as np
from sklearn import svm


def truncate_by_dist(dataset, labels, colors, maxdist, C, eps, svc=None):
    if svc is None:
        svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
    assert np.dot(svc.coef_[0], svc.coef_[0]) + svc.intercept_**2 > 0
    assert len(svc.support_) > 0
    dataset_trunc = []
    labels_trunc = []
    colors_trunc = []
    indices = []
    for i in range(len(labels)):
        if dist(dataset[i], list(svc.coef_[0]), svc.intercept_) < maxdist:
            dataset_trunc.append(dataset[i])
            labels_trunc.append(labels[i])
            colors_trunc.append(colors[i])
            indices.append(i)
    assert len(dataset_trunc) > 0
    n_t, m = np.shape(dataset_trunc)
    eps_t = eps*len(dataset)/n_t
    svc1 = svm.SVC(kernel='linear', C=C).fit(dataset_trunc, labels_trunc)
    w_trunc = list(svc1.coef_[0])
    b_trunc = svc1.intercept_
    # generating pilot h
    h = np.zeros(n_t*m)
    for i in range(n_t):
        for j in range(m):
            h[n_t*j+i] = np.sign(np.dot(w_trunc, dataset_trunc[i])+b_trunc) * \
                         (w_trunc[j]/np.sqrt(np.dot(w_trunc, w_trunc)))*(np.sqrt(eps_t))
    # generating vector of Lagrange multipliers
    l_trunc = []
    for i in range(len(dataset_trunc)):
        if i in svc1.support_:
            l_trunc.append(svc1.dual_coef_[0][list(svc1.support_).index(i)] / labels_trunc[i])
        elif labels_trunc[i] * (np.dot(w_trunc, dataset_trunc[i]) + b_trunc) < 0:
            l_trunc.append(C)
        else:
            l_trunc.append(0.0)
    a_trunc = [1 - min(1, labels_trunc[i] * (np.dot(w_trunc, dataset_trunc[i]) + b_trunc)) for i in range(len(dataset_trunc))]
    x_trunc = w_trunc + list(b_trunc) + list(h) + l_trunc + a_trunc
    return dataset_trunc, labels_trunc, colors_trunc, indices, x_trunc, eps_t


def dist(point, w, b):
    assert len(point) == len(w)
    return abs(np.dot(w, point)+b)/np.sqrt(sum(a**2 for a in w))


def truncate_get_support(dataset, labels, colors, C, eps, svc=None):
    if svc is None:
        svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
    assert np.dot(svc.coef_[0], svc.coef_[0]) + svc.intercept_**2 > 0
    assert len(svc.support_) > 0
    dataset_trunc = []
    labels_trunc = []
    colors_trunc = []
    indices = list(svc.support_)
    for i in indices:
        dataset_trunc.append(dataset[i])
        labels_trunc.append(labels[i])
        colors_trunc.append(colors[i])
    n_t, m = np.shape(dataset_trunc)
    eps_t = eps*len(dataset)/n_t
    svc1 = svm.SVC(kernel='linear', C=C).fit(dataset_trunc, labels_trunc)
    w_trunc = list(svc1.coef_[0])
    b_trunc = svc1.intercept_
    # generating pilot h
    h = np.zeros(n_t*m)
    for i in range(n_t):
        for j in range(m):
            h[n_t*j+i] = np.sign(np.dot(w_trunc, dataset_trunc[i])+b_trunc) * \
                         (w_trunc[j]/np.sqrt(np.dot(w_trunc, w_trunc)))*(np.sqrt(eps_t))
    # generating vector of Lagrange multipliers
    l_trunc = []
    for i in range(len(dataset_trunc)):
        if i in svc1.support_:
            l_trunc.append(svc1.dual_coef_[0][list(svc1.support_).index(i)] / labels_trunc[i])
        elif labels_trunc[i] * (np.dot(w_trunc, dataset_trunc[i]) + b_trunc) < 0:
            l_trunc.append(C)
        else:
            l_trunc.append(0.0)
    a_trunc = [1 - min(1, labels_trunc[i] * (np.dot(w_trunc, dataset_trunc[i]) + b_trunc)) for i in range(len(dataset_trunc))]
    x_trunc = w_trunc + list(b_trunc) + list(h) + l_trunc + a_trunc
    return dataset_trunc, labels_trunc, colors_trunc, indices, x_trunc, eps_t