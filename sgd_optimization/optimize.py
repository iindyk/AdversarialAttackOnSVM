import random
import numpy as np


def sgd_adv_class_sbs(dataset_full, labels_full, batch_size=-1, maxit=100, precision=1e4):
    nit = 0
    m = len(dataset_full[0])
    if batch_size == -1:
        batch_size = int(len(dataset_full)/10.0)
    while nit < maxit:
        indices = random.sample(range(0, len(dataset_full)), batch_size)
        dataset = [dataset_full[i] for i in indices]
        labels = [labels_full[i] for i in indices]
        #solve adv
        #solve class
        nit+=1