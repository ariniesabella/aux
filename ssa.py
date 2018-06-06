from pylab import *
import math
import matplotlib.pyplot as plt
import random
import time
import os
import pandas as pd
from consts import *

from auxiliary import omega, propensity, react
from determ import determ_solver_for_hybrid_model


def _gillespie_ssa(q, s, X):
    # tau = 0.0001
    t = 0
    tau_arr = [0]
    tau_arr2 = []
    k = 0
    tau = 0
    step1 = True
    props = np.array([[propensity(0, X[:, k])], [0], [0], [0], [0], [0], [0], [propensity(0, X[:, k])], [0], [0], [0], [0], [0], [0], [0]])
    while (True):
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        alpha0 = 0
        pr_tmp = []
        for i in range(q):
            p = propensity(i, X[:, k])
            pr_tmp.append([p])
            alpha0 += p
        if not step1:
            props = np.hstack((props, pr_tmp))
        else:
            step1 = False
        tau = pow(alpha0, -1) * math.log(pow(r1, -1),)
        if tau + t > T:
            break
        tmp = 0
        J = [0 for _ in range(q)]
        for i in range(q):
            if r2 >= tmp and \
                            r2 < (tmp + pow(alpha0, -1) * propensity(i, X[:, k])):
                J[i] = 1
            tmp += pow(alpha0, -1) * propensity(i, X[:, k])
        x_tmp = react(J, X[:, k])
        X = np.hstack((X, x_tmp))

        t += tau
        tau_arr.append(t)
        tau_arr2.append(tau)
        k += 1

    return X, tau_arr, props


# ___________________
#
#  hybrid
# ___________________

def _count_tau(q, k, X, step0, params):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    alpha0 = 0
    pr_tmp = []
    if step0:
        pr_tmp = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0]])
        alpha0 = 0
        tau = 0.01
        return tau, pr_tmp, r1, r2, alpha0
    for i in range(q):
        p = propensity(i, X[:, k], params)
        pr_tmp.append([p])
        alpha0 += p

    return pow(alpha0, -1) * math.log(pow(r1, -1),), pr_tmp, r1, r2, alpha0


def _direct_hybrid(path_to_folder, params, X0_init):
    X_stoch = X0_init
    X_det = X_det0
    t = 0
    tau_arr = [0]
    tau_arr2 = []
    k = 0
    tau = 0
    step0 = True
    x_det_cur = X_det[:, k]
    props = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0]])
    while 1:
        tau, pr_tmp, r1, r2, alpha0 = _count_tau(q, k, X_stoch, step0, params)
        step0 = False
        if tau + t > T:
            break
        consts = [X_stoch[7, k], X_stoch[10, k]]
        x_det_cur_new = determ_solver_for_hybrid_model(t, t + tau, x_det_cur, consts, params)
        for kk, xxx in enumerate(x_det_cur_new):
            if xxx < 0.0:
                x_det_cur_new[kk] = 0.0

        # reset values of stoch (a, r)
        X_stoch[3, k] = x_det_cur_new[0]
        X_stoch[1, k] = r0 - x_det_cur_new[2] - x_det_cur_new[4]

        # reset values of stoch (cL, mL, mS)
        X_stoch[2, k] = x_det_cur_new[4]
        X_stoch[0, k] = x_det_cur_new[5]
        X_stoch[8, k] = x_det_cur_new[6]

        tmp = 0
        J = [0 for _ in range(q)]

        tau_, pr_tmp, r1, r2, alpha0 = _count_tau(q, k, X_stoch, step0, params)
        if tau_ < 0:
            break

        props = np.hstack((props, pr_tmp))

        for i in range(q):
            if r2 >= tmp and \
                            r2 < (tmp + pow(alpha0, -1) * propensity(i, X_stoch[:, k], params)):
                J[i] = 1
            tmp += pow(alpha0, -1) * propensity(i, X_stoch[:, k], params)
        x_tmp = react(J, X_stoch[:, k])


        for kk, xxx in enumerate(x_tmp):
            if xxx[0] < 0:
                x_tmp[kk] = [0.0]

        X_stoch = np.hstack((X_stoch, x_tmp))

        t += tau
        tau_arr.append(t)
        tau_arr2.append(tau)

        file = open(path_to_folder + 'time.log', 'a')
        file.write("%s\n" % tau)
        file.close()

        # reset values of det
        x_det_cur = x_det_cur_new
        x_det_cur[0] = X_stoch[3, k]
        x_det_cur[4] = X_stoch[2, k]
        x_det_cur[5] = X_stoch[0, k]
        x_det_cur[6] = X_stoch[8, k]

        x_tmp_det = []
        for z in range(len(x_det_cur_new)):
            x_tmp_det.append([x_det_cur_new[z]])
        X_det = np.hstack((X_det, x_tmp_det))

        ################# acceleration for the param search
        #if X_stoch[7, k] > L0+10 or X_stoch[10, k] > S0+10:
        #    break
        ################
        #if tau < 0.00000001:
        #    break
        k += 1

    return X_stoch, X_det, tau_arr, props


def new_hybrid_test():

    number_of_runs = 8

    # path_to_pics = "/home/alex/PycharmProjects/stoch/"
    path_to_pics = '/home/alex/Desktop/data2/params/'

    par_names = ['lambda_m^L', 'lambda_m^S', 'lambda_p^L', 'lambda_p^S']

    countS = 0
    countL = 0

    params = [0.001, 10000, 1, 50]
    params0 = [0.0, 0.0001, 0.01, 1, 10, 100, 1000]
    print(params0)
    num_of_params = len(params0)

    dictS = {k: [] for k in params0}
    dictL = {k: [] for k in params0}
    for i in range(num_of_params):
        for l in range(number_of_runs):
            #params = [100*(l + 2), 1000*(l + 2), 10, 10]
            params[0] = params0[i]

            adds = ''
            cur_path = ("l_m_L=" + str(params[0])
                        + "_l_m_S=" + str(params[1]) + "_l_p_L=" +
                        str(params[2]) + "_l_p_S=" + str(params[3]))
            path = path_to_pics + cur_path + "/"
            if not os.path.exists(path):
                os.makedirs(path)

            start_time = time.time()
            X_stoch_new, X_det_new, tau_arr, props = _direct_hybrid(path, params, X0)
            end_time = time.time()
            print(end_time - start_time)

            dL = X_stoch_new[7, -1] - L0
            dS = X_stoch_new[10, -1] - S0

            dictS[params0[i]].append(dS)
            dictL[params0[i]].append(dL)
            '''
            new_plot_fig(i+1, tau_arr, X_stoch_new, X_det_new, path, props, adds)
            file = open(path + 'time.log', 'a')
            file.write(str(end_time - start_time) + '\n')
            file.close()
            file = open(path + 'param_values.log', 'w+')
            for w in range(4):
                file.write(par_names[w] + '=' + str(params[w])+'\n')
            file.close()
            '''
    return dictS, dictL, path_to_pics


def new_hybrid_param_search():

    number_of_runs = 10

    # path_to_pics = "/home/alex/PycharmProjects/stoch/"
    path_to_pics = '/home/alex/Desktop/data/new_hybrid/real_param_search/counts/'

    par_names = ['lambda_m^L', 'lambda_m^S', 'lambda_p^L', 'lambda_p^S']

    countS = 0
    countL = 0

    params = [0.001, 10000, 1, 50]
    #lam_pLs = np.linspace(0.55, 0.6, 2)
    lam_pLs = [1]
    dLs = []
    dSs = []
    for index, pL in enumerate(lam_pLs):
        dL = 0
        dS = 0
        params[2] = pL
        adds = ''
        cur_path = ("l_m_L=" + str(params[0])
                    + "_l_m_S=" + str(params[1]) + "_l_p_L=" +
                    str(params[2]) + "_l_p_S=" + str(params[3]))
        path = path_to_pics + cur_path + "/"
        if not os.path.exists(path):
            os.makedirs(path)

        for l in range(number_of_runs):
            start_time = time.time()
            X_stoch_new, X_det_new, tau_arr, props = _direct_hybrid(path, params, X0)
            end_time = time.time()
            print(end_time - start_time)

            dL += (X_stoch_new[7, -1] - L0)/number_of_runs
            dS += (X_stoch_new[10, -1] - S0)/number_of_runs

            new_plot_fig(index + 1, tau_arr, X_stoch_new, X_det_new, path, props, adds)

            file = open(path + 'time.log', 'a')
            file.write(str(end_time - start_time) + '\n')
            file.close()

        dLs.append(dL)
        dSs.append(dS)
        file = open(path + 'param_values.log', 'w+')
        for w in range(4):
            file.write(par_names[w] + '=' + str(params[w])+'\n')
        file.close()
    file = open(path_to_pics + 'fraction.txt', 'w')
    for item in dLs:
        file.write("%s\n" % item)
    for item in dSs:
        file.write("%s\n" % item)
    file.close()

    return path_to_pics




def new_plot_fig(z, tau_arr, X_stoch_new, X_det_new, path, props, adds):
    col = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    for i in range(s):
        plt.figure(str(z) + var_names[i])
        if i == 7:
            plt.plot(tau_arr, X_stoch_new[i, :] - L0, color=col)
        elif i == 10:
            plt.plot(tau_arr, X_stoch_new[i, :] - S0, color=col)
        else:
            plt.plot(tau_arr, X_stoch_new[i, :], color=col)
        plt.grid(True)
        plt.xlabel("min")
        plt.ylabel(var_names[i])
        if not os.path.exists(path + "species/"):
            os.makedirs(path + "species/")
        plt.savefig(path + "species/" + var_names[i] + adds)

    for i in range(N):
        plt.figure(str(z) + det_var_names[i])
        plt.plot(tau_arr, X_det_new[i, :], color=col)
        plt.grid(True)
        plt.xlabel("min")
        plt.ylabel(det_var_names[i])

        plt.savefig(path + "species/det_" + det_var_names[i] + adds)

    for j in range(q):
        plt.figure(str(z) + "Reaction_" + str(j))
        plt.plot(tau_arr, props[j, :], color=col)
        plt.grid(True)
        plt.xlabel("min")
        plt.ylabel("propensity of reaction # " + str(j))
        if not os.path.exists(path + "reactions/"):
            os.makedirs(path + "reactions/")
        plt.savefig(path + "reactions/Reaction_" + str(j) + adds)


def change_init_cond():
    number_of_runs = 1
    params = [0.001, 10000, 1, 50]
    adds = "_L0"

    cur_path = ("l_m_L=" + str(params[0])
                + "_l_m_S=" + str(params[1]) + "_l_p_L=" +
                str(params[2]) + "_l_p_S=" + str(params[3]) + "_incond_change_L0S0")
    count = 0
    if not os.path.exists(cur_path):
        os.makedirs(cur_path)
    path_to_pics = "/home/alex/Desktop/data2/"  # /new_hybrid/real_param_search/init_cond/"
    path = path_to_pics + cur_path + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    L0s = np.linspace(200, 400, 3)
    S0s = np.linspace(26486, 52972, 3)

    dLs = []
    dSs = []
    das = []
    drs = []

    for index1, l in enumerate(L0s):
        for index2, ss in enumerate(S0s):
            X0_init = np.array([[0], [r0], [0], [a0], [0], [0], [0], [l], [0], [0], [ss]])
            dL = 0
            dS = 0
            da = 0
            dr = 0
            adds = "_L0=" + str(int(l)) + "_S0=" + str(int(ss))
            print(adds)

            success_runs = number_of_runs
            for _ in range(number_of_runs):
                start_time = time.time()
                try:
                    X_stoch_new, X_det_new, tau_arr, props = _direct_hybrid(path, params, X0_init)
                except:
                    print("Unexpected error with computing: ", sys.exc_info()[0])
                    success_runs -= 1
                    continue
                end_time = time.time()
                print('Time: ', end_time - start_time)
                #try:
                #    new_plot_fig(index + 1000, tau_arr, X_stoch_new, X_det_new, path, props, adds)
                #except:
                #    print("Unexpected error with plot: ", sys.exc_info()[0])
                dL += (X_stoch_new[7, -1] - l)
                dS += (X_stoch_new[10, -1] - S0)
                da += X_stoch_new[3, -1]
                dr += X_stoch_new[1, -1]

            dL /= success_runs
            dS /= success_runs
            da /= success_runs
            dr /= success_runs

            with open(path + "L_L0S0.txt", "a") as myfile1:
                myfile1.write(str(dL) + '\n')
            with open(path + "S_L0S0.txt", "a") as myfile2:
                myfile2.write(str(dS) + '\n')
            with open(path + "a_L0S0.txt", "a") as myfile3:
                myfile3.write(str(da) + '\n')
            with open(path + "r_L0S0.txt", "a") as myfile4:
                myfile4.write(str(dr) + '\n')

'''

    adds = "_L0"

    cur_path = ("l_m_L=" + str(params[0])
                + "_l_m_S=" + str(params[1]) + "_l_p_L=" +
                str(params[2]) + "_l_p_S=" + str(params[3]) + "_incond_change_L0")
    count = 0
    if not os.path.exists(cur_path):
        os.makedirs(cur_path)
    path_to_pics = "/home/alex/Desktop/data2/"#/new_hybrid/real_param_search/init_cond/"
    path = path_to_pics + cur_path + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    L0s = np.linspace(400, 400, 1)


    dLs = []
    dSs = []
    das = []
    drs = []

    for index, l in enumerate(L0s):
        X0_init = np.array([[0], [r0], [0], [a0], [0], [0], [0], [l], [0], [0], [S0]])
        dL = 0
        dS = 0
        da = 0
        dr = 0
        adds = "_L0="+str(int(l))
        print(adds)

        if index == 0:
            tmp = number_of_runs
        else:
            tmp = number_of_runs
        success_runs = number_of_runs
        for _ in range(tmp):
            start_time = time.time()
            try:
                X_stoch_new, X_det_new, tau_arr, props = _direct_hybrid(path, params, X0_init)
            except:
                print("Unexpected error with computing: ", sys.exc_info()[0])
                success_runs -= 1
                continue
            end_time = time.time()
            print('Time: ', end_time - start_time)
            try:
                new_plot_fig(index + 1000, tau_arr, X_stoch_new, X_det_new, path, props, adds)
            except:
                print("Unexpected error with plot: ", sys.exc_info()[0])
            dL += (X_stoch_new[7, -1] - l)
            dS += (X_stoch_new[10, -1] - S0)
            da += X_stoch_new[3, -1]
            dr += X_stoch_new[1, -1]

        dL /= success_runs
        dS /= success_runs
        da /= success_runs
        dr /= success_runs

        dLs.append(dL)
        dSs.append(dS)
        drs.append(da)
        das.append(dr)
    
    data = pd.DataFrame(
        {'L0': L0s,
         'dL': dLs,
         'dS': dSs,
         'da': das,
         'dr': drs
        })
    data.to_csv(path + 'data_L0.csv', sep='\t')

    cur_path = ("l_m_L=" + str(params[0])
                + "_l_m_S=" + str(params[1]) + "_l_p_L=" +
                str(params[2]) + "_l_p_S=" + str(params[3]) + "_incond_change_S0")

    if not os.path.exists(cur_path):
        os.makedirs(cur_path)

    path = path_to_pics + cur_path + "/"
    if not os.path.exists(path):
        os.makedirs(path)


    S0s = np.linspace(66217, 92701, 3)

    dLs = []
    dSs = []
    das = []
    drs = []

    for index, ss in enumerate(S0s):
        X0_init = np.array([[0], [r0], [0], [a0], [0], [0], [0], [L0], [0], [0], [ss]])
        dL = 0
        dS = 0
        da = 0
        dr = 0
        adds = "_S0=" + str(int(ss))
        success_runs = number_of_runs
        for _ in range(number_of_runs):
            start_time = time.time()
            try:
                X_stoch_new, X_det_new, tau_arr, props = _direct_hybrid(path, params, X0_init)
            except:
                print("Unexpected error with computing: ", sys.exc_info()[0])
                success_runs -= 1
                continue
            end_time = time.time()
            print('Time: ', end_time - start_time)
            try:
                new_plot_fig(index + 50, tau_arr, X_stoch_new, X_det_new, path, props, adds)
            except:
                print("Unexpected error with plot: ", sys.exc_info()[0])



            dL += (X_stoch_new[7, -1] - L0)
            dS += (X_stoch_new[10, -1] - ss)
            da += X_stoch_new[3, -1]
            dr += X_stoch_new[1, -1]

        dL /= success_runs
        dS /= success_runs
        da /= success_runs
        dr /= success_runs

        dLs.append(dL)
        dSs.append(dS)
        drs.append(da)
        das.append(dr)

    data = pd.DataFrame(
        {'S0': S0s,
         'dL': dLs,
         'dS': dSs,
         'da': das,
         'dr': drs
         })
    data.to_csv(path + 'data_S0.csv', sep='\t')
'''