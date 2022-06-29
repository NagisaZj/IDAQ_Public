import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

from plot_utils import *
if __name__ =="__main__":
    mine_data=data_read(paths=['./outputfin2/walker-rand-params/2021_03_25_08_45_38/progress.csv',
                  './outputfin2/walker-rand-params/2021_03_25_08_45_40/progress.csv'])

    mine_data2 = data_read(paths=['./outputfin2/walker-rand-params/2021_03_25_08_46_48/progress.csv',
                                         './outputfin2/walker-rand-params/2021_03_25_08_46_56/progress.csv'])

    #print(maml_data[0][-1],maml_data[1][-1])
    datas = [mine_data,mine_data2]
    legends = ['MetaCURE', 'PEARL']
    # datas = [mine_data_new_intr, promp_data, erl2_data, mame_data]
    # legends = ['MetaCURE', 'ProMP', 'E-RL^2', 'MAME']
    plot_all(datas, legends, 0)
    plt.title('Cheetah-Vel-Sparse', size=30)
    # plt.plot(mine_data_new_intr[0], np.ones(mine_data_new_intr[0].shape) * 9.98, color='olive', linestyle='--',
    #         linewidth=2, label='EPI')
    # legend()
    plt.show()
