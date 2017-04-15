import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from random import uniform

dataset = []
labels = []
colors = []
for i in range(0, 600):
    x1 = uniform(0, 200)
    x2 = uniform(0, 200)
    dataset.append([x1, x2])
    if x2 >= x1-10:
        labels.append(1)
        colors.append((1, 0, 0))
    else:
        labels.append(-1)
        colors.append((0, 0, 1))

C = 1.0  # SVM regularization parameter
h = 1 # step size in the mesh
svc = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
x_min, x_max = 0, 200
y_min, y_max = 0, 200
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('linear svm')

plt.show()