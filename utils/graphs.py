import matplotlib.pyplot as plt
import numpy as np
import utils.ofs.obj_con_functions_v1 as of1


def graph_results(A, B, eps, dataset, dataset_infected, inf_points, colors, x, n_t, svc_orig, svc_inf):
    m = len(dataset[0])
    w, b, h, l, a = of1.decompose_x(x, m, n_t)
    plt.subplot(321)
    plt.title('original')
    plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
    plt.subplot(322)
    plt.title('infected')
    plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors,
                cmap=plt.cm.coolwarm)
    plt.subplot(323)
    step = (B - A) / 100.0  # step size in the mesh

    x_min, x_max = int(A - 2 * (eps / m) ** 0.5), int(B + 2 * (eps / m) ** 0.5)
    y_min, y_max = int(A - 2 * (eps / m) ** 0.5), int(B + 2 * (eps / m) ** 0.5)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = np.sign([i[0] * w[0] + i[1] * w[1] + b for i in np.c_[xx.ravel(), yy.ravel()]])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors,
                cmap=plt.cm.coolwarm)
    plt.title('opt on inf data')
    plt.plot([i[0] for i in inf_points], [i[1] for i in inf_points], 'go', mfc='none')
    plt.subplot(324)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = svc_inf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors,
                cmap=plt.cm.coolwarm)
    plt.plot([i[0] for i in inf_points], [i[1] for i in inf_points], 'go', mfc='none')
    plt.title('inf svc on inf data')

    plt.subplot(326)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = svc_inf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
    plt.title('inf svc on orig data')

    plt.subplot(325)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = svc_orig.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
    plt.title('orig svc on orig data')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()