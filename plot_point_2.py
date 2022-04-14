import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

def data_read(paths=['./outputfin2/cheetah-vel-sparse/2019_11_20_08_52_39/progress.csv',
                  '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_20_16_01_14/progress.csv',
                  '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_19_19_57_40/progress.csv']):
    mine_values = []
    num_trajs = len(paths)
    mine_paths = paths
    shortest = 10000000000
    for p in mine_paths:
        csv_data = pd.read_csv(p)
        values_steps = csv_data['Number of env steps total'].values
        values_returns = csv_data['AverageReturn_all_test_tasks'].values
        # values_returns = smoothingaverage(values_returns)
        values_returns = smooth(values_returns[None],10)[0]
        #print(values_steps.shape)
        length = values_steps.shape[0]
        shortest = length if length < shortest else shortest
        mine_values.append([values_steps,values_returns])
        '''plots = csv.reader(csvfile,delimiter=',')
        print(plots)
        for row in plots:
            print(row)'''

    xs = mine_values[0][0][:shortest]/1e6
    ys = np.zeros([shortest,num_trajs])
    for i  in range(num_trajs):
        ys[:,i] = mine_values[i][1][:shortest]
    mean = np.mean(ys,1)
    std = np.std(ys,1)
    return xs,mean,std

def data_read_2(paths=['./outputfin2/cheetah-vel-sparse/2019_11_20_08_52_39/progress.csv',
                  '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_20_16_01_14/progress.csv',
                  '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_19_19_57_40/progress.csv']):
    mine_values = []
    num_trajs = len(paths)
    mine_paths = paths
    shortest = 10000000000
    for p in mine_paths:
        csv_data = pd.read_csv(p)
        values_steps = csv_data['Number of env steps total'].values
        values_returns = csv_data['AverageReturn_all_test_tasks_online'].values
        # values_returns = smoothingaverage(values_returns)
        values_returns = smooth(values_returns[None],10)[0]
        #print(values_steps.shape)
        length = values_steps.shape[0]
        shortest = length if length < shortest else shortest
        mine_values.append([values_steps,values_returns])
        '''plots = csv.reader(csvfile,delimiter=',')
        print(plots)
        for row in plots:
            print(row)'''

    xs = mine_values[0][0][:shortest]/1e6
    ys = np.zeros([shortest,num_trajs])
    for i  in range(num_trajs):
        ys[:,i] = mine_values[i][1][:shortest]
    mean = np.mean(ys,1)
    std = np.std(ys,1)
    return xs,mean,std

def plot_full(data,color,name):
    plt.plot(data[0], data[1], color,label=name)
    plt.fill_between(data[0], data[1] - data[2], data[1] + data[2], color=color, alpha=0.2)
    # plt.plot(data[0], np.ones(data[0].shape) * np.mean(data[1][-100:]), color+':')


def smoothingaverage(data,window_size=5):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data,window,'same')

def smooth(data, smooth_range):
	print('hhhhhhh', type(data), len(data))
	new_data = np.zeros_like(data)
	for i in range(0, data.shape[-1]):
		if i < smooth_range:
			new_data[:, i] = 1. * np.sum(data[:, :i + 1], axis=1) / (i + 1)
		else:
			new_data[:, i] = 1. * np.sum(data[:, i - smooth_range + 1:i + 1], axis=1) / smooth_range

	return new_data

if __name__ =="__main__":
    mine_data=data_read(paths=['/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed3/progress.csv',
                  '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed4/progress.csv',
                  '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed5/progress.csv'])
    mine_data_2 = data_read_2(paths=['/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed3/progress.csv',
                                 '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed4/progress.csv',
                                 '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed5/progress.csv'])

    mine_data = data_read(paths=['/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed6/progress.csv',
                                 '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed7/progress.csv',
                                 '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed8/progress.csv'])
    mine_data_2 = data_read_2(
        paths=['/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed6/progress.csv',
               '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed7/progress.csv',
               '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed8/progress.csv'])
    mine_data = data_read(paths=['/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed9/progress.csv',
                                 '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed10/progress.csv',
                                 '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed11/progress.csv'])
    mine_data_2 = data_read_2(
        paths=['/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed9/progress.csv',
               '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed10/progress.csv',
               '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed11/progress.csv'])

    #print(maml_data[0][-1],maml_data[1][-1])
    plt.figure()
    plt.xlabel('Million Environment Samples', size=20)
    plt.ylabel('Average Return', size=20)

    plot_full(mine_data,'b','FOCAL')
    plot_full(mine_data_2, 'r', 'FOCAL Online')


    #plt.plot(pearl_data[0], pearl_data[1], 'b')
    #plt.fill_between(pearl_data[0], pearl_data[1] - pearl_data[2], pearl_data[1] + pearl_data[2], color='b', alpha=0.2)
    #plt.plot(pearl_data[0], np.ones(pearl_data[0].shape) * np.max(pearl_data[1]), 'b:')
    plt.legend()
    plt.show()
