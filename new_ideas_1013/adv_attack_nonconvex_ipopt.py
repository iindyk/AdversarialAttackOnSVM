import pyipopt
import datetime
import numpy as np
import utils.ofs.obj_con_functions_v1 as of1
from sklearn import svm
from sklearn.metrics import accuracy_score
from utils.datasets_parsers.random_dataset_generator import generate_random_dataset as grd
from utils.graphs import graph_results as graph


n = 5  # training set size
m = 2  # features
C = 1.0  # SVM regularization parameter
flip_size = 0  # random attack size
A = 0  # left end of interval for generating points
B = 1  # right end of interval for generating points
eps = 0.1*(B-A)  # upper bound for (norm of h)**2


dataset, labels, colors = grd(n=n, m=m, a=A, b=B, attack=flip_size, read=False, write=False, sep='linear')
svc_orig = svm.SVC(kernel='linear', C=C).fit(dataset, labels)
x0, x_L, x_U, g_L, g_U, err_orig = of1.get_initial_data(dataset, labels, C, eps, svc_orig)

nvar = m + 1 + n * (m + 2)  # number of variables
ncon = 3*n+m+2  # number of constraints
nnzj = ncon*nvar  # number of nonzero elements in Jacobian of constraints function
nnzh = nvar**2  # number of nonzero elements in Hessian of Lagrangian


def eval_f(x):
    assert len(x) == nvar
    return of1.adv_obj(x, dataset, labels)


def eval_grad_f(x):
    assert len(x) == nvar
    return of1.adv_obj_gradient(x, dataset, labels)


def eval_g(x):
    assert len(x) == nvar
    return np.append(of1.class_constr_inf_eq_nonconvex(x, dataset, labels, C),
                     of1.class_constr_inf_ineq_nonconvex(x, dataset, labels, eps))


def eval_jac_g(x, flag):
    if flag:
        i_s = []
        j_s = []
        for i in range(ncon):
            for j in range(nvar):
                i_s.append(i)
                j_s.append(j)
        return np.array(i_s), np.array(j_s)
    else:
        assert len(x) == m + 1 + n * (m + 2)
        jac = np.append(of1.class_constr_inf_eq_nonconvex_jac(x, dataset, labels, C),
                        of1.class_constr_inf_ineq_nonconvex_jac(x, dataset, labels, eps),
                        axis=0)
        assert np.shape(jac) == (ncon, nvar)
        ret = []
        for i in range(ncon):
            for j in range(nvar):
                ret.append(jac[i, j])
        return np.array(ret)


print(of1.class_constr_inf_eq_nonconvex_jac(x0, dataset, labels, C))
nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f,
                     eval_grad_f, eval_g, eval_jac_g)
nlp.str_option("derivative_test", "first-order")
nlp.str_option('derivative_test_print_all', 'yes')
nlp.num_option('derivative_test_perturbation', 1e-8)
nlp.int_option('max_iter', 3000)
nlp.int_option('print_frequency_iter', 100)

print(datetime.datetime.now(), ": Going to call solve")
x_opt, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
nlp.close()
print('status: ', status)

# analysis of results:
w, b, h, l, a = of1.decompose_x(x_opt, m, n)
dataset_infected = []
print('attack norm= '+str(np.dot(h, h)/n))
print('objective value= '+str(obj))
k = 0
for i in range(0, n):
    tmp = []
    for j in range(0, m):
        tmp.append(dataset[i][j]+h[j*n+k])
    dataset_infected.append(tmp)
# define infected points for graph
inf_points = []
k = 0
for i in range(n):
    if sum([h[j*n+k]**2 for j in range(m)]) > 0.9*eps:
        inf_points.append(dataset_infected[i])
    k += 1
svc_inf = svm.SVC(kernel='linear', C=C)
svc_inf.fit(dataset_infected, labels)
predicted_labels_inf_svc = svc_inf.predict(dataset)
err_inf_svc = 1 - accuracy_score(labels, predicted_labels_inf_svc)
print('err on infected dataset by svc is '+str(int(100*err_inf_svc))+'%')
predicted_labels_inf_opt = np.sign([np.dot(dataset[i], w)+b for i in range(0, n)])
err_inf_opt = 1 - accuracy_score(labels, predicted_labels_inf_opt)
print('err on infected dataset by opt is '+str(int(100*err_inf_opt))+'%')
graph(A, B, eps, dataset, dataset_infected, inf_points, colors, x_opt, n, svc_orig, svc_inf)

