import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from random import randint

from datetime import datetime

from read_data_libsvm import read_data_libsvm
from read_data_uci_ionosphere import read_data_uci

print("start at "+ str(datetime.now()))
ts = read_data_uci("data_uci_ionosphere.txt")
# handling missing features
'''imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(ts)
imp.transform(ts)
'''
n = len(ts)  # number of points
m = len(ts[0]) - 1  # number of features
training_points = []
training_labels = []
test_points = []
test_labels = []
test_indices = []
# choosing indices for test set
while len(test_indices) < int(n/3):
    ri = randint(0, n)
    if ri not in test_indices:
        test_indices.append(ri)

for i in range(0, n):
    if i in test_indices:
        test_points.append(ts[i][1:m+2])
        test_labels.append(ts[i][0])
    else:
        training_points.append(ts[i][1:m + 2])
        training_labels.append(ts[i][0])



# stochastic SVM
def objective(x):
    f = 0.0
    for l in range(0, len(training_points)):
        temp = 0.0
        for k in range(0, m):
            temp += training_points[l][k]*x[k]
        f += min(1.0, training_labels[l]*(temp + x[m]))  # x[m]=b
    return float(-f/n)
x0 = np.asarray([1.0 for p in range(0, m+1)])
solution = minimize(objective, x0, bounds=None)
w = solution.x[0:m]
b = solution.x[m]
predicted_labelsS_training = np.sign([np.dot(training_points[i], w)+b for i in range(0, len(training_points))])
predicted_labelsS_test = np.sign([np.dot(test_points[i], w)+b for i in range(0, len(test_points))])
errS_training = 1 - accuracy_score(training_labels, predicted_labelsS_training)
errS_test = 1 - accuracy_score(test_labels, predicted_labelsS_test)
print("Stochastic SVM error training = "+str(errS_training)+" test = " + str(errS_test))


# Soft-margin C-SVM with linear kernel
C = 1.0  # regularization parameter
svcL = svm.SVC(kernel='linear', C=C).fit(training_points, training_labels)
predicted_labelsL_training = svcL.predict(training_points)
predicted_labelsL_test = svcL.predict(test_points)
errL_training = 1 - accuracy_score(training_labels, predicted_labelsL_training)
errL_test = 1 - accuracy_score(test_labels, predicted_labelsL_test)
print("Soft-margin C-SVM with linear kernel training error = "+str(errL_training)+" test = " + str(errL_test))
'''
# Soft-margin C-SVM with polynomial kernel
C = 1.0  # regularization parameter
degree = 3  # degree of polynomial kernel
svcP = svm.SVC(kernel='poly', degree=degree, C=C).fit(training_points, training_labels)
predicted_labelsP = svcP.predict(test_points)
errP = 1 - accuracy_score(test_labels, predicted_labelsP)
print("Soft-margin C-SVM with polynomial kernel error = "+str(errP))

# Soft-margin C-SVM with rbf kernel
C = 1.0
svcR = svm.SVC(kernel='rbf', C=C).fit(training_points, training_labels)
predicted_labelsR = svcR.predict(test_points)
errR = 1 - accuracy_score(test_labels, predicted_labelsR)
print("Soft-margin C-SVM with rbf kernel error = "+str(errR))

# Soft-margin C-SVM with sigmoid kernel
C = 1.0
svcR = svm.SVC(kernel='sigmoid', C=C).fit(training_points, training_labels)
predicted_labelsG = svcR.predict(test_points)
errG = 1 - accuracy_score(test_labels, predicted_labelsG)
print("Soft-margin C-SVM with sigmoid kernel error = "+str(errG))
'''

# Nu-SVM with linear kernel
nu = 0.5
svc = svm.NuSVC(nu=nu, kernel='linear').fit(training_points, training_labels)
predicted_labelsN_training = svc.predict(training_points)
predicted_labelsN_test = svc.predict(test_points)
errN_training = 1 - accuracy_score(training_labels, predicted_labelsN_training)
errN_test = 1 - accuracy_score(test_labels, predicted_labelsN_test)
print("Nu-SVM with linear kernel training error = "+str(errN_training)+" test error = " + str(errN_test))
