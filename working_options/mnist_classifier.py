import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

from utils.datasets_parsers.read_mnist import read

C = 5.0
gamma = 0.05
digits = [3, 8]
path = '/home/iindyk/PycharmProjects/AdversarialAttackOnSVM/datasets/MNIST'
training_all = list(read(dataset='training', path=path))
test_all = list(read(dataset='testing', path=path))
training_data = []
training_labels = []
# reformatting and choosing only 2 classes for binary classification
for dl in training_all:
    label, pixels = dl
    if label in digits:
        training_data.append(np.reshape(pixels, -1)/255)
        training_labels.append(label)
del training_all
print('training set size= '+str(len(training_labels)))

test_data = []
test_labels = []
for dl in test_all:
    label, pixels = dl
    if label in digits:
        test_data.append(np.reshape(pixels, -1)/255)
        test_labels.append(label)
del test_all
print('test set size= '+str(len(test_labels)))
svc = svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(training_data, training_labels)
err = 1 - accuracy_score(test_labels, svc.predict(test_data))
print('test error = '+str(err))