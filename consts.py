import numpy as np

# stochastic variable names list
var_names = ["mL", "r", "cL", "a", "O1", "O2", "bL", "L", "mS", "bS", "S"]


# constant values

w_q = 948.93
theta_nr = 4.38
d_m = 0.1
d_p = 0.00135
k_b = 0.00333
k_u = 1
gamma_max = 1260
n_L = 6000
n_S = 300
K_gamma = 21000
chi_max = 840
K_chi = 324000
K_L = 0.0028
r0 = pow(10, 7)

# initial values
L0 = 100  # 106414 * 0.01
S0 = 1324285 * 0.01
a0 = 100000000
X0 = np.array([[0], [r0], [0], [a0], [0], [0], [0], [L0], [0], [0], [S0]])

#X_det0 = np.array([[a0], [0], [0], [0]])
#new
X_det0 = np.array([[a0], [0], [0], [0], [0], [0], [0]])

#NEW q and N!!!!
q = 9  # number of stochastic reactions
s = 11  # number of stochastic species
N = 7  # number of deterministic species

T = 40

########################
#params = [0, 1, 0, 1]
########################

# deterministic variable names list
#det_var_names = ["a", "q", "cq", "mq"]
# new deterministic variable names list
det_var_names = ["a", "q", "cq", "mq", "cL", "mL", "mS"]

A = pow(10, 9)
n_q = 431
alpha_ = 0.0042
beta_ = 5.96 * pow(10, -9)

# state change matrix
nu = np.zeros((11, 15), dtype=int)
nu[0][0] = 1

nu[0][1] = -1

nu[0][2] = -1
nu[1][2] = -1
nu[2][2] = 1

nu[0][3] = 1
nu[1][3] = 1
nu[2][3] = -1

nu[0][4] = 1
nu[1][4] = 1
nu[2][4] = -1
nu[3][4] = (-1)*n_L
nu[4][4] = 1
nu[5][4] = 1

nu[0][5] = -1
nu[5][5] = -1
nu[6][5] = 1

nu[3][6] = (-1)*n_L
nu[6][6] = -1
nu[7][6] = 1

nu[8][7] = 1

nu[8][8] = -1

nu[8][9] = -1
nu[5][9] = -1
nu[9][9] = 1

nu[3][10] = (-1)*n_S
nu[9][10] = -1
nu[10][10] = 1

nu[4][11] = -1

nu[5][12] = -1

nu[6][13] = -1

nu[9][14] = -1

# tau leap constants
eps = 0.03
