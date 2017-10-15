import matplotlib.pyplot as plt
import numpy as np
import utils.ofs.obj_con_functions_v1 as of1
from sklearn.metrics import accuracy_score


def graph_results(A, B, eps, dataset,labels, dataset_infected, inf_points, colors, x, n, svc_orig, svc_inf):
    m = len(dataset[0])
    w, b, h, l, a = of1.decompose_x(x, m, n)
    step = (B - A) / 100.0  # step size in the mesh
    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.2, 1.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    plt.subplot(221)
    plt.title('original dataset')
    plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
    plt.subplot(222)
    plt.title('infected dataset')
    plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors,
                cmap=plt.cm.coolwarm)
    plt.plot([i[0] for i in inf_points], [i[1] for i in inf_points], 'go', mfc='none')
    '''plt.subplot(323)
    Z = np.sign([i[0] * w[0] + i[1] * w[1] + b for i in np.c_[xx.ravel(), yy.ravel()]])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors,
                cmap=plt.cm.coolwarm)
    plt.title('opt on inf data')
    plt.plot([i[0] for i in inf_points], [i[1] for i in inf_points], 'go', mfc='none')
    plt.subplot(324)
    Z = svc_inf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors,
                cmap=plt.cm.coolwarm)
    plt.title('inf svc on inf data')'''

    predicted_labels_inf_svc = svc_inf.predict(dataset)
    err_inf_svc = 1 - accuracy_score(labels, predicted_labels_inf_svc)

    plt.subplot(224)
    Z = svc_inf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
    plt.title('inf svc on orig data, err='+str(int(100*err_inf_svc))+'%')

    predicted_labels_orig_svc = svc_orig.predict(dataset)
    err_orig_svc = 1 - accuracy_score(labels, predicted_labels_orig_svc)

    plt.subplot(223)
    Z = svc_orig.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
    plt.title('orig svc on orig data, err='+str(int(100*err_orig_svc))+'%')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()