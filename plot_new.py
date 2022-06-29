import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

from plot_utils import *
if __name__ =="__main__":
    mine_testing_data = \
	    data_read(paths=['output/sparse-point-robot__2022-06-28_23-15-44/sparse-point-robot/debug/progress.csv'],
	              load_name='AverageReturn_all_test_tasks')

    mine_training_context_data = \
	    data_read(paths=['output/sparse-point-robot__2022-06-28_23-15-44/sparse-point-robot/debug/progress.csv'],
	              load_name='AverageTrainReturn_all_train_tasks')

    mine_training_data = \
	    data_read(paths=['output/sparse-point-robot__2022-06-28_23-15-44/sparse-point-robot/debug/progress.csv'],
	              load_name='AverageReturn_all_train_tasks')

    #print(maml_data[0][-1],maml_data[1][-1])
    datas = [mine_testing_data, mine_training_context_data, mine_training_data]
    legends = ['CPEARL-testing', 'CPEARL-training-context', 'CPEARL-training']
    # datas = [mine_data_new_intr, promp_data, erl2_data, mame_data]
    # legends = ['MetaCURE', 'ProMP', 'E-RL^2', 'MAME']
    plot_all(datas, legends, 0)
    plt.title('Sparse-Point-Robot', size=30)
    # plt.plot(mine_data_new_intr[0], np.ones(mine_data_new_intr[0].shape) * 9.98, color='olive', linestyle='--',
    #         linewidth=2, label='EPI')
    # legend()
    plt.show()
