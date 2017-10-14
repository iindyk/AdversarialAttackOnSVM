import pyipopt
import numpy as np
import utils.ofs.obj_con_functions_v1 as of1
from sklearn import svm
from utils.datasets_parsers.random_dataset_generator import generate_random_dataset as grd


n = 20  # training set size
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
nnzh = 0  # number of nonzero elements in Hessian of objective function


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
        assert np.shape(jac) == ncon, nvar
        ret = []
        for i in range(ncon):
            for j in range(nvar):
                ret.append(jac[i, j])
        return np.array(ret)

