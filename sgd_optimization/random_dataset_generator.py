import numpy as np
from numpy.random import uniform
import pickle


def generate_random_dataset(n=200, m=2, a=0, b=100, attack=0, read=False, write=False, sep='linear'):
    if not read:
        if sep not in ('linear', 'quad'):
            raise Exception('No such separation function: ' + sep)
        if sep == 'linear':
            def sep_value(x):
                return sum(x) - m * (b - a) / 2.0
        else:
            def sep_value(x):
                return sum([l**2 for l in x]) - m * ((b - a)**2 / 4.0)
        dataset = uniform(a, b, (n, m))
        labels = []
        colors = []
        for i in range(0, n):
            if sep_value(dataset[i]) >= 0:
                labels.append(1.0)
                colors.append((1, 0, 0))
            else:
                labels.append(-1.0)
                colors.append((0, 0, 1))
        # random attack
        for i in range(0, attack):
            if labels[i] == 1.0:
                labels[i] = -1.0
                colors[i] = (0, 0, 1)
            else:
                labels[i] = 1.0
                colors[i] = (1, 0, 0)
        if write:
            dataset_pickle = {"dataset": dataset, "labels": labels, "colors": colors}
            pickle.dump(dataset_pickle,
                        open("/home/iindyk/PycharmProjects/AdversarialAttackOnSVM/datasets/random.pickle", "wb+"),
                        pickle.HIGHEST_PROTOCOL)
    else:
        print('reading data saved data, ignoring all dataset parameters')
        try:
            dataset_pickle = \
                pickle.load(open("/home/iindyk/PycharmProjects/AdversarialAttackOnSVM/datasets/random.pickle", "rb"))
            dataset = dataset_pickle.get("dataset")
            labels = dataset_pickle.get("labels")
            colors = dataset_pickle.get("colors")
        except Exception as e:
            print("Unable to read data: ", e)
            return
    return dataset, labels, colors
