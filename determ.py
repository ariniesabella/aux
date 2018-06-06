
# ___________________
#
# deterministic part
# ___________________

from scipy.integrate import ode
from pylab import *

from consts import *
from auxiliary import omega
import matplotlib.pyplot as plt
import random
import os



const_L = L0
const_S = S0
params_0 = 0
params_1 = 0
params_2 = 0
params_3 = 0

def lambda_R():
    return alpha_ + beta_ * (const_L + const_S)


def nu_q(a, c_q):
    return (gamma_max / n_q) * a * c_q / (K_gamma + a)


def nu_L(a, c_L):
    return (gamma_max * 3 / n_L) * a * c_L / (K_gamma + a)


def nu_int_L(a, b_L, O1):
    return (chi_max / n_L) * (K_L * O1 / (1 + O1 * K_L)) * a * b_L / (K_chi + a)


def nu_int_S(a, b_S):
    return (chi_max / n_S) * a * b_S / (K_chi + a)

#par_names = ['lambda_m^L', 'lambda_m^S', 'lambda_p^L', 'lambda_p^S']

################################
###___Deterministic_model____###
################################


def deterministic_sys(t, X):
    f = [A - nu_q(X[0], X[2]) - nu_L(X[0], X[4]) - nu_int_L(X[0], X[9], X[7]) - nu_int_S(X[0], X[11]) \
            - lambda_R() * X[0],
         nu_q(X[0], X[2]) - d_p * X[1],
         k_b * (r0 - X[2] - X[4]) * X[3] - k_u * X[2] - nu_q(X[0], X[2]) - d_p * X[2],
         omega(X[0]) - d_m * X[3] + nu_q(X[0], X[2]) - k_b * (r0 - X[2] - X[4]) * X[3] + k_u * X[2],
         k_b * (r0 - X[2] - X[4]) * X[5] - k_u * X[4] - (params_2 + d_p) * X[4],
         - k_b * (r0 - X[2] - X[4]) * X[5] + k_u * X[4] + omega(X[0]) * const_L / L0 - (d_m + params_0) * X[5] - \
                k_b * X[5] * X[8] -  nu_q(X[0], X[2]),
         omega(X[0]) * const_S / S0 - (d_m + params_1) * X[6] - k_b * X[6] * X[8],
         nu_L(X[0], X[4]) - (d_m + params_2) * X[7],
         nu_L(X[0], X[4]) - (d_m + params_2) * X[8] - k_b * X[8] * X[5] - k_b * X[8] * X[6],
         k_b * X[8] * X[5] - nu_int_L(X[0], X[9], X[7]) - (d_m + params_2) * X[9],
         nu_int_L(X[0], X[9], X[7]),
         k_b * X[8] * X[6] - nu_int_S(X[0], X[11]) - (d_m + params_2) * X[11],
         nu_int_S(X[0], X[11])
         ]
    return f


def determ_solver(t_start, t_end, init_cond, params):
    NN = 10  # number of steps

    global params_0
    params_0 = params[0]
    global params_1
    params_1 = params[1]
    global params_2
    params_2 = params[2]
    global params_3
    params_3 = params[3]

    solver = ode(deterministic_sys)

    t = []
    sol = []
    solver.set_initial_value(init_cond, t_start)
    solver.set_integrator('lsoda')

    dt = (t_end - t_start) / NN

    while solver.successful() and solver.t < t_end:
        solver.integrate(solver.t + dt)
        t.append(solver.t)
        sol.append(solver.y)

    return array(solver.y)


def determ_test(params):
    X_determ = np.array([[a0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [L0], [0], [S0]])
    x_det_cur = X_determ[:, 0]
    t = 0.0
    tau = 0.01
    k = 0
    t_arr = []
    while 1:
        t_arr.append(t)
        if tau + t > T:
            break
        x_det_cur = determ_solver(t, t + tau, x_det_cur, params)

        for kk, xxx in enumerate(x_det_cur):
            if xxx < 0.0:
                x_det_cur[kk] = 0.0

        t += tau

        x_tmp_det = []
        for z in range(len(x_det_cur)):
            x_tmp_det.append([x_det_cur[z]])
        X_determ = np.hstack((X_determ, x_tmp_det))

        k += 1

    return X_determ, t_arr


def plot_determ_model_sol(path_to_folder, params):

    X_determ, t_arr = determ_test(params)

    col = (0.1, 0.2, 0.5)
    det_model_var_names = ["a", "q", "cq", "mq", "cL", "mL", "mS", "O1", "O2", "bL", "L", "bS", "S"]
    number_of_vars = len(det_model_var_names)
    for i in range(number_of_vars):
        plt.figure(det_model_var_names[i])
        plt.plot(t_arr, X_determ[i, :], color=col)
        plt.grid(True)
        plt.ylim(0.0, 1.05 * max(X_determ[i, :]))
        plt.xlabel("min")
        plt.ylabel(det_model_var_names[i])
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)
        plt.savefig(path_to_folder + det_model_var_names[i])


################################
###_______Hybrid_model______###
################################


def deterministic_sys_for_hybrid_model(t, X):
    f = [A - nu_q(X[0], X[2]) - lambda_R() * X[0],
         nu_q(X[0], X[2]) - d_p * X[1],
         k_b * (r0 - X[2] - X[4]) * X[3] - k_u * X[2] - nu_q(X[0], X[2]) - d_p * X[2],
         omega(X[0]) - d_m * X[3] + nu_q(X[0], X[2]) - k_b * (r0 - X[2] - X[4]) * X[3] + k_u * X[2],
         k_b * (r0 - X[2] - X[4]) * X[5] - k_u * X[4] - (params_2 + d_p) * X[4],
         - k_b * (r0 - X[2] - X[4]) * X[5] + k_u * X[4] + omega(X[0]) * const_L / L0 - (d_m + params_0) * X[5],
         omega(X[0]) * const_S / S0 - (d_m + params_1) * X[6]
         ]
    return f


def determ_solver_for_hybrid_model(t_start, t_end, init_cond, consts, params):
    NN = 10

    global const_L
    const_L = consts[0]
    global const_S
    const_S= consts[1]

    global params_0
    params_0 = params[0]
    global params_1
    params_1 = params[1]
    global params_2
    params_2 = params[2]


    solver = ode(deterministic_sys_for_hybrid_model)

    t = []
    sol = []
    solver.set_initial_value(init_cond, t_start)
    solver.set_integrator('lsoda')

    dt = (t_end - t_start) / NN

    while solver.successful() and solver.t < t_end:
        solver.integrate(solver.t + dt)
        t.append(solver.t)
        sol.append(solver.y)

    return array(solver.y)
