import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

training_set = np.genfromtxt("data_clean.txt", delimiter=" ")

print(str(training_set[0]))