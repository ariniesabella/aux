#from ssa import _hybrid_test
from consts import *
import numpy as np
from ssa import new_hybrid_test, new_hybrid_param_search,change_init_cond
from determ import plot_determ_model_sol

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

import pandas as pd

def draw_plot(data, edge_color, fill_color, x_label, y_label, path, adds):

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True, vert=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color, alpha=0.6)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.xticks([i + 1 for i in range(len(data.index))], list(data.index))
    plt.grid()
    plt.savefig(path  + adds)
    #plt.show()

if __name__ == '__main__':
    #_hybrid_test()
    #np.savetxt('test.out', nu.astype(int), delimiter=',')
    #hybrid_tau_leap_test()

    #new_hybrid_test()


    #df = pd.DataFrame(np.random.rand(10, 5))
    #df = df.transpose()
    #print(df)

    #draw_plot(df, 'black', 'lightgreen', 'l', 'k')
    #lam_pLs = np.linspace(0.1, 1, 10)
    #print(lam_pLs)
    #path = new_hybrid_param_search()
    #change_init_cond()

    dictS, dictL, path_to_pics = new_hybrid_test()
    dfS = pd.DataFrame.from_dict(dictS)
    dfL = pd.DataFrame.from_dict(dictL)

    dfS = dfS.transpose()
    dfL = dfL.transpose()
    adds = 'lambda_m_L'#'lambda_m_L'
    addsL = 'L__' + adds
    addsS = 'S__' + adds

    dfS.to_csv(path_to_pics + "params/" + addsS + '.txt', sep='\t')
    dfL.to_csv(path_to_pics + "params/" + addsL + '.txt', sep='\t')

    draw_plot(dfS, 'black', 'lightgreen', r'$\lambda_m^L$', r'$\Delta$S', path_to_pics + "params/", addsS )
    draw_plot(dfL, 'black', 'cyan', r'$\lambda_m^L$', r'$\Delta$L',  path_to_pics + "params/", addsL)

    '''
    path_to_folder = "/home/alex/Desktop/data2/determ/"
    params = [0.001, 10000, 1, 50]
    plot_determ_model_sol(path_to_folder, params)
    '''



