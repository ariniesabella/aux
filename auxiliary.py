from consts import *


def omega(a):
    return w_q * a / (theta_nr + a)


#new
def propensity(i, x, params):
    if i == 0:
        return gamma_max * 3 * x[3] * x[2] / (n_L * (K_gamma + x[3]))
    if i == 1:
        return k_b * x[5] * x[0]
    if i == 2:
        f = K_L * x[4] / (1 + K_L * x[4])
        return chi_max * f * x[3] * x[6] / (n_L * (K_chi + x[3]))

    if i == 3:
        return k_b * x[5] * x[8]
    if i == 4:
        return chi_max * x[3] * x[9] / (n_S * (K_chi + x[3]))
    if i == 5:
        return (params[2] + d_m) * x[4]
    if i == 6:
        return (params[2] + d_m) * x[5]
    if i == 7:
        return (params[2] + d_m) * x[6]
    if i == 8:
        return (params[3] + d_m) * x[9]


def create_this_abomination_omg(x):
    tmp = []
    for i in range(len(x)):
        tmp.append([x[i]])
    return tmp


def react(J, X):
    if J[0]:
        X[0] += 1
        X[1] += 1
        X[2] -= 1
        X[3] -= n_L  # is that so?
        X[4] += 1
        X[5] += 1
    if J[1]:
        X[0] -= 1
        X[5] -= 1
        X[6] += 1
    if J[2]:
        X[3] -= n_L
        X[6] -= 1
        X[7] += 1

    if J[3]:
        X[8] -= 1
        X[5] -= 1
        X[9] += 1
    if J[4]:
        X[3] -= n_S
        X[9] -= 1
        X[10] += 1
    if J[5]:
        X[4] -= 1
    if J[6]:
        X[5] -= 1
    if J[7]:
        X[6] -= 1
    if J[8]:
        X[9] -= 1

    return create_this_abomination_omg(X)

# tau leap


def d_prop(i, j, x, params):
    if j == 0:
        if i == 3:
            return (x[7] * w_q / L0) * theta_nr / ((theta_nr + x[3])*(theta_nr + x[3]))
        if i == 7:
            return omega(x[3]) / L0
        else:
            return 0.0
    if j == 1:
        if i == 0:
            return params[0] + d_m
    if j == 2:
        if i == 0:
            return k_b * x[1]
        if i == 1:
            return k_b * x[0]
        return k_b * x[1] * x[0]
    if j == 3:
        if i == 2:
            return k_u
    if j == 4:
        if i == 2:
            return gamma_max * 3 * x[3] / (n_L * (K_gamma + x[3]))
        if i == 3:
            return gamma_max * 3 * x[3] * x[2] / (n_L * (K_gamma + x[3]) * (K_gamma + x[3]))
    if j == 5:
        if i == 0:
            return k_b * x[5]
        if i == 5:
            return k_b * x[0]
    if j == 6:
        if i == 4:
            f = 1 / ((1 + K_L * x[4]) * (1 + K_L * x[4]))
            return K_L * chi_max * f * x[3] * x[6] / (n_L * (K_chi + x[3]))
        if i == 3:
            f = K_L * x[4] / (1 + K_L * x[4])
            return chi_max * f * K_chi * x[6] / (n_L * (K_chi + x[3]) * (K_chi + x[3]))
        if i == 6:
            f = K_L * x[4] / (1 + K_L * x[4])
            return chi_max * f * x[3] * x[6] / (n_L * (K_chi + x[3]))
    if j == 7:
        if i == 10:
            return omega(x[3]) / S0
        if i == 3:
            return w_q * x[10] * theta_nr / (S0 * (theta_nr + x[3]) * (theta_nr + x[3]))
    if j == 8:
        if i == 8:
            return params[1] + d_m
    if j == 9:
        if i == 5:
            return k_b * x[8]
        if i == 8:
            return k_b * x[5]
    if j == 10:
        if i == 9:
            return chi_max * x[3] / (n_S * (K_chi + x[3]))
        if i == 3:
            return chi_max * K_chi * x[9] / (n_S * (K_chi + x[3])*(K_chi + x[3]))
    if j == 11:
        if i == 4:
            return params[2] + d_m
    if j == 12:
        if i == 5:
            return params[2] + d_m
    if j == 13:
        if i == 6:
            return params[2] + d_m
    if j == 14:
        if i == 9:
            return params[3] + d_m
    return 0.0


def f(j, k, x):
    tmp = 0
    for i in range(s):
        tmp += d_prop(i, j, x) * nu[i][k]
    return tmp

def sigma_mu(j, x):
    tmp_s = 0
    tmp_m = 0
    for k in range(q):
        tt = f(j, k, x)
        tmp_m += tt * propensity(k, x)
        tmp_s += tt * tt * propensity(k, x)
    return abs(tmp_m), tmp_s

def count_alpha0(x):
    alpha0 = 0
    for i in range(q):
        p = propensity(i, x)
        alpha0 += p

    return alpha0


def count_tau_from_leap_cond(X, iter):
    tmp = []
    xx = X[:, iter]
    a0 = count_alpha0(xx)
    for j in range(q):
        m, sgm = sigma_mu(j, xx)
        tmp.append(min(eps * a0 / float(m), eps * a0 * eps * a0 / float(sgm)))

    print(tmp)
    return min(tmp)


