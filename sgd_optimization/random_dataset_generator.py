import numpy as np
from numpy.random import uniform
import pickle


def generate_random_dataset(n=200, m=2, a=0, b=100, attack=0, read=False, write=False):
    if not read:
        dataset = uniform(a, b, (n, m))
        labels = np.array([])
        colors = []
        for i in range(0, n):
            if sum(dataset[i]) >= m * (b - a) / 2.0:
                np.append(labels, 1.0)
                colors.append((1, 0, 0))
            else:
                np.append(labels, -1.0)
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
                        open("C:\\Users\\HP\\PycharmProjects\\AdversarialAttackOnSVM\\datasets\\random.pickle", "wb+"),
                        pickle.HIGHEST_PROTOCOL)
    else:
        try:
            dataset_pickle = \
                pickle.load(open("C:\\Users\\HP\\PycharmProjects\\AdversarialAttackOnSVM\\datasets\\random.pickle", "rb"))
            dataset = dataset_pickle.get("dataset")
            labels = dataset_pickle.get("labels")
            colors = dataset_pickle.get("colors")
        except Exception as e:
            print("Unable to read data: ", e)
            return
    return dataset, labels, colors
