import numpy as np
from sklearn import svm


def truncate(dataset, labels, maxdist, svc=None):
    dataset_trunc = []
    labels_trunc = []
    w = list(svc.coef_[0])
    b = svc.intercept_





    x_trunc = []
    return dataset_trunc, labels_trunc, x_trunc