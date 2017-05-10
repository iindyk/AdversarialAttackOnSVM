import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


assets = np.array([[0.004, -0.025, 0.009, 0.012, 0.047, 0.006, -0.019, -0.037, 0.025, 0.021, 0.017, 0.019],
[0.014, 0, -0.039, 0.016, -0.006, -0.021, 0.07, -0.022, 0.019, 0.025, 0.054, 0.04],
[0.001, 0.006, 0.005, 0.019, 0.016, -0.052, 0.057, 0.027, 0.039, 0, 0.011, 0.002],
[-0.012, -0.021, 0.062, 0.036, -0.002, 0.015, -0.038, -0.003, 0.024, 0.012, 0.048, -0.007],
[-0.043, 0.005, 0.023, 0, 0.023, 0.034, 0.04, 0.029, -0.013, -0.04, 0.011, 0.003],
[0.015, -0.027, -0.01, -0.027, 0.002, 0.056, 0.038, -0.004, 0.08, 0.001, 0.013, 0.026],
[-0.001, 0.011, 0.056, -0.024, 0.019, -0.015, -0.048, 0.019, 0.062, 0.023, 0.002, -0.017],
[0.039, 0.03, 0.003, -0.004, 0.016, 0.003, -0.021, 0.018, -0.026, -0.022, 0.026, 0.073],
[0.017, 0.02, -0.024, -0.004, 0.019, -0.03, 0.039, 0.025, 0.021, 0.054, -0.011, 0.056],
[0.108, -0.003, 0.061, 0.008, 0.024, -0.013, -0.037, 0.053, -0.009, -0.021, 0.026, -0.009]])

n = len(assets[0])  # number of scenarios
m = len(assets)  # number of assets
p = 1.0/n  # probability of scenario
alpha = 0.3  # parameter of AVaR
capital = 100000  # initial capital
cs = [0.25, 0.5, 0.75]  # parameters for parts b) and c)
q = [1 for j in range(0, m)]  # assets prices
w0 = [10000 for k in range(0, m)]  # initial guess for optimization
# w - weight list - result
means = []  # mean values of optimal assets (for graph)
devs = []  # deviations of optimal assets (for graph)
colors = []  # colors of points for graph


# constraint on capital function
def constraint(w):
    return capital - np.dot(w, q)
con = {'type': 'ineq', 'fun': constraint}
cons = ([con])
bounds = [(0, capital) for j in range(0, m)]  # bound on weight of each asset


def objective_for_avar(x, w):
    temp = 0
    for i in range(0, n):
        temp += p*max(0, - np.dot(assets[:, i], w)-x)
    return x + temp/alpha


def avar(w):
    x0 = 1
    sol = minimize(objective_for_avar, x0, args=w, bounds=None, method='Nelder-Mead')
    if not sol.success:
        print(sol.message)
    return sol.fun


#  part a
sol_a = minimize(avar, w0, bounds=bounds, constraints=cons)
if not sol_a.success:
    print(sol_a.message)
# print(sol_a.nit)
optimal_asset_a = np.array([np.dot(sol_a.x, assets[:, j]) for j in range(0, n)])
print("part a) solution(weight vector) is "+str(sol_a.x) + "; use of capital = " + str(np.sum(sol_a.x)))
print("mean of optimal asset = " + str(np.mean(optimal_asset_a)) + " stdev = " + str(np.std(optimal_asset_a)))
means.append(np.mean(optimal_asset_a))
devs.append(np.std(optimal_asset_a))
colors.append((0, 1, 0))


for c in cs:
    #  part b
    def objective_b(w):
        return -(1-c)*np.mean(np.array([np.dot(w, assets[:, j]) for j in range(0, n)])) + c*avar(w)
    sol_b = minimize(objective_b, w0, bounds=bounds, constraints=cons)
    if not sol_b.success:
        print(sol_b.message)
    # print(sol_b.nit)
    optimal_asset_b = np.array([np.dot(sol_b.x, assets[:, j]) for j in range(0, n)])
    print("part b), c = "+str(c)+" solution(weight vector) is "+str(sol_b.x) + "; use of capital = "+str(np.sum(sol_b.x)))
    print("mean of optimal asset = " + str(np.mean(optimal_asset_b)) + " stdev = " + str(np.std(optimal_asset_b)))
    means.append(np.mean(optimal_asset_b))
    devs.append(np.std(optimal_asset_b))
    colors.append((0, 0, 1))

    # part c
    def lower_semi_dev(w):
        temp = 0
        mean = np.mean(np.array([np.dot(w, assets[:, j]) for j in range(0, n)]))
        for i in range(0, n):
            temp += p*max(0, mean - np.dot(w,assets[:, i]))
        return temp


    def objective_c(w):
        return -np.mean(np.array([np.dot(w, assets[:, j]) for j in range(0, n)])) + c*lower_semi_dev(w)

    sol_c = minimize(objective_c, w0, bounds=bounds, constraints=cons)
    if not sol_c.success:
        print(sol_c.message)
    # print(sol_c.nit)
    optimal_asset_c = np.array([np.dot(sol_c.x, assets[:, j]) for j in range(0, n)])
    print("part c), c = "+str(c)+" solution(weight vector) is "+str(sol_c.x) + "; use of capital = "+str(np.sum(sol_c.x)))
    print("mean of optimal asset = " + str(np.mean(optimal_asset_c)) + " stdev = " + str(np.std(optimal_asset_c)))
    means.append(np.mean(optimal_asset_c))
    devs.append(np.std(optimal_asset_c))
    colors.append((1, 0, 0))

part_a = plt.scatter([means[0]], [devs[0]], c=[colors[0]])
part_b = plt.scatter([means[2*i+1] for i in range(0, 3)], [devs[2*i+1] for i in range(0, 3)],
                     c=[colors[2*i+1] for i in range(0, 3)])
part_c = plt.scatter([means[2*i+2] for i in range(0, 3)], [devs[2*i+2] for i in range(0, 3)],
                     c=[colors[2*i+2] for i in range(0, 3)])
plt.legend([part_a, part_b, part_c], ['part a)', 'part b)', 'part c)'])
plt.xlabel('mean')
plt.ylabel('variance')
# annotation
for c in cs:
    plt.annotate('c=' + str(c), (means[1+2*cs.index(c)], devs[1+2*cs.index(c)]))
    plt.annotate('c=' + str(c), (means[2+2*cs.index(c)], devs[2+2*cs.index(c)]))
plt.show()