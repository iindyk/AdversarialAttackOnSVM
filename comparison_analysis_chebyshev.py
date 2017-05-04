import numpy as np
import read_data_uci as uci
from scipy.optimize import minimize
from sklearn import svm
from sklearn.metrics import accuracy_score
from random import randint


ts = uci.read_data_uci_survival("data_uci_survival.txt")

n = len(ts)  # number of points
m = len(ts[0]) - 1  # number of features
training_points = []
training_labels = []
for i in range(0, n):
        training_points.append(ts[i][1:m + 2])
        training_labels.append(ts[i][0])


# stochastic SVM
def objective_stoch(x):
    f = 0.0
    for l in range(0, len(training_points)):
        temp = 0.0
        for k in range(0, m):
            temp += training_points[l][k]*x[k]
        f += min(1.0, training_labels[l]*(temp + x[m]))  # x[m]=b
    return float(-f/len(training_points))
x0 = np.asarray([1.0 for p in range(0, m+1)])
options = {'maxiter': 700}
solution_stoch = minimize(objective_stoch, x0, bounds=None, method='SLSQP', options=options)
w_stoch = solution_stoch.x[0:m]
b_stoch = solution_stoch.x[m]
print(solution_stoch.success)
print("opt value is "+str(solution_stoch.fun))
print(objective_stoch(solution_stoch.x))
print(solution_stoch.message)
print(solution_stoch.nit)


#  Chebyshev's inequality SVM - deviation
def objective_cheb(x):
    av = 0.0
    for l in range(0, len(training_points)):
        temp = 0.0
        for k in range(0, m):
            temp += training_points[l][k]*x[k]
        av += (training_labels[l]*(temp + x[m]))  # x[m]=b
    return -av/len(training_points)


def constr_cheb_dev(x):
    dev = 0.0
    av = 0.0
    for l in range(0, len(training_points)):
        temp = 0.0
        for k in range(0, m):
            temp += training_points[l][k] * x[k]
        av += (training_labels[l] * (temp + x[m]))/len(training_points)
        dev += ((-training_labels[l] * (temp + x[m]) + av) ** 2)/len(training_points)  # x[m]=b
    return 1 - dev


con_dev = {'type': 'ineq', 'fun': constr_cheb_dev}
cons_dev = ([con_dev])

solution_cheb_dev = minimize(objective_cheb, x0, bounds=None, constraints=cons_dev, options=options)

print(solution_cheb_dev.fun)
print(solution_cheb_dev.success)
print(solution_cheb_dev.message)
print(solution_cheb_dev.nit)

w_cheb_dev = solution_cheb_dev.x[0:m]
b_cheb_dev = solution_cheb_dev.x[m]


#  Chebyshev's inequality SVM - semi-deviation
def constr_cheb_semi(x):
    semidev = 0.0
    av = 0.0
    for l in range(0, len(training_points)):
        temp = 0.0
        for k in range(0, m):
            temp += training_points[l][k] * x[k]
        av += (training_labels[l] * (temp + x[m]))/len(training_points)
        semidev += (max(0.0, (-training_labels[l] * (temp + x[m]) + av)) ** 2)/len(training_points)  # x[m]=b
    return 1 - semidev


con_semi = {'type': 'ineq', 'fun': constr_cheb_semi}
cons_semi = ([con_semi])
solution_cheb_semi = minimize(objective_cheb, x0, bounds=None, constraints=cons_semi, options=options)
print(solution_cheb_semi.fun)
print(solution_cheb_semi.success)
print(solution_cheb_semi.message)
print(solution_cheb_semi.nit)

w_cheb_semi = solution_cheb_semi.x[0:m]
b_cheb_semi = solution_cheb_semi.x[m]

# Soft-margin C-SVM with linear kernel
C = 1.0  # regularization parameter
svcL = svm.SVC(kernel='linear', C=C).fit(training_points, training_labels)


# tests
errS = []
errCD = []
errCS = []
errC = []
for i in range(0, 100):
    test_points = []
    test_labels = []
    test_indices = []  # choosing indices for test set
    while len(test_indices) < int(n/3):  # test sample size
        ri = randint(0, n)
        if ri not in test_indices:
            test_indices.append(ri)
    for j in range(0, n):
        if j in test_indices:
            test_points.append(ts[j][1:m + 2])
            test_labels.append(ts[j][0])
    #  stoch
    predicted_labelsS_test = np.sign([np.dot(test_points[i], w_stoch)+b_stoch for i in range(0, len(test_points))])
    errS.append(1 - accuracy_score(test_labels, predicted_labelsS_test))
    #  cheb dev
    predicted_labelsCD_test = np.sign(
        [np.dot(test_points[i], w_cheb_dev)+b_cheb_dev for i in range(0, len(test_points))])
    errCD.append(1 - accuracy_score(test_labels, predicted_labelsCD_test))
    #  cheb semi-dev
    predicted_labelsCS_test = np.sign(
        [np.dot(test_points[i], w_cheb_semi) + b_cheb_semi for i in range(0, len(test_points))])
    errCS.append(1 - accuracy_score(test_labels, predicted_labelsCS_test))
    #  soft-margin
    predicted_labelsC_test = svcL.predict(test_points)
    errC.append(1 - accuracy_score(test_labels, predicted_labelsC_test))

print(errS)
print(errCD)
print(errCS)
print(errC)
print("Stochastic SVM avg error = "+str(np.mean(errS))+"; deviation = " + str(np.std(errS)))
print("Cheb dev avg error = "+str(np.mean(errCD))+"; deviation = " + str(np.std(errCD)))
print("Cheb semi avg error = "+str(np.mean(errCS))+"; deviation = " + str(np.std(errCS)))
print("Soft-margin avg error = "+str(np.mean(errC))+"; deviation = " + str(np.std(errC)))