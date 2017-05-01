import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer

from read_data_libsvm import read_data_libsvm
from read_data_uci import read_data_uci

ts = read_data_uci("data_uci_liver_disorder.txt")
n = len(ts)  # number of points
m = len(ts[0]) - 1  # number of features
training_points = []
training_labels = []
test_points = training_points
test_labels = training_labels
for ar in ts:
    training_labels.append(ar[0])
    training_points.append(ar[1:3])
# handling missing features
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(test_points)
imp.transform(test_points)


# stochastic SVM
def objective(x):
    f = 0.0
    for l in range(0, n):
        temp = 0.0
        for k in range(0, m):
            temp += training_points[l][k]*x[k]
        f += min(1.0, training_labels[l]*(temp + x[m]))  # x[m]=b
    return float(-f/n)
x0 = np.asarray([1.0 for p in range(0, m+1)])
solution = minimize(objective, x0, bounds=None)
w = solution.x[0:m]
b = solution.x[m]
predicted_labelsS = np.sign([np.dot(test_points[i], w)+b for i in range(0, n)])
errS = 1 - accuracy_score(test_labels, predicted_labelsS)
print("Stochastic SVM error = "+errS)


# Soft-margin C-SVM with linear kernel
C = 1.0  # regularization parameter
svcL = svm.SVC(kernel='linear', C=C).fit(training_points, training_labels)
predicted_labelsL = svcL.predict(test_points)
errL = 1 - accuracy_score(test_labels, predicted_labelsL)
print("Soft-margin C-SVM with linear kernel error = "+errL)

# Soft-margin C-SVM with polynomial kernel
C = 1.0  # regularization parameter
degree = 3  # degree of polynomial kernel
svcP = svm.SVC(kernel='poly', degree=degree, C=C).fit(training_points, training_labels)
predicted_labelsP = svcP.predict(test_points)
errP = 1 - accuracy_score(test_labels, predicted_labelsP)
print("Soft-margin C-SVM with polynomial kernel error = "+errP)

# Soft-margin C-SVM with rbf kernel
C = 1.0
svcR = svm.SVC(kernel='rbf', C=C).fit(training_points, training_labels)
predicted_labelsR = svcR.predict(test_points)
errR = 1 - accuracy_score(test_labels, predicted_labelsR)
print("Soft-margin C-SVM with rbf kernel error = "+errR)

# Soft-margin C-SVM with sigmoid kernel
C = 1.0
svcR = svm.SVC(kernel='sigmoid', C=C).fit(training_points, training_labels)
predicted_labelsG = svcR.predict(test_points)
errG = 1 - accuracy_score(test_labels, predicted_labelsG)
print("Soft-margin C-SVM with sigmoid kernel error = "+errG)


# Nu-SVM with rbf kernel
nu = 0.5
svc = svm.NuSVC(nu=nu, kernel='rbf').fit(training_points, training_labels)
predicted_labelsN = svc.predict(test_points)
errN = 1 - accuracy_score(test_labels, predicted_labelsN)
print("Nu-SVM with rbf kernel error = "+errN)
