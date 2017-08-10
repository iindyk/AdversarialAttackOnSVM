import numpy as np
from sklearn import svm


def truncate(dataset, labels, colors, maxdist, C, svc=None):
    if svc is None:
        svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
    dataset_trunc = []
    labels_trunc = []
    colors_trunc = []
    for i in range(len(labels)):
        if dist(dataset[i], list(svc.coef_[0]), svc.intercept_) < maxdist:
            dataset_trunc.append(dataset[i])
            labels_trunc.append(labels[i])
            colors_trunc.append(colors[i])
    assert len(dataset_trunc) > 0
    svc1 = svm.SVC(kernel='linear', C=C).fit(dataset_trunc, labels_trunc)
    w_trunc = list(svc1.coef_[0])
    b_trunc = svc1.intercept_
    l_trunc = []
    for i in range(len(dataset_trunc)):
        if i in svc1.support_:
            l_trunc.append(svc1.dual_coef_[0][list(svc1.support_).index(i)] / labels_trunc[i])
        elif labels_trunc[i] * (np.dot(w_trunc, dataset_trunc[i]) + b_trunc) < 0:
            l_trunc.append(C)
        else:
            l_trunc.append(0.0)
    a_trunc = [1 - min(1, labels_trunc[i] * (np.dot(w_trunc, dataset_trunc[i]) + b_trunc)) for i in range(len(dataset_trunc))]
    x_trunc = w_trunc + list(b_trunc) + list(np.zeros(len(dataset_trunc)*len(dataset_trunc[0]))) + l_trunc + a_trunc
    return dataset_trunc, labels_trunc, colors_trunc, x_trunc


def dist(point, w, b):
    assert len(point) == len(w)
    return abs(np.dot(w, point)+b)/np.sqrt(sum(a**2 for a in w)+b**2)

print(dist([1, 1],))