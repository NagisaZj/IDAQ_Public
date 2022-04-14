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
                  '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_19_19_57_40/progress.csv'],key='Oracle_Prediction_Loss'):
    mine_values = []
    num_trajs = len(paths)
    mine_paths = paths
    shortest = 10000000000
    for p in mine_paths:
        csv_data = pd.read_csv(p)
        values_steps = csv_data['Number of env steps total'].values
        values_returns = csv_data[key].values
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

    pp = ['/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed9/progress.csv',
               '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed10/progress.csv',
               '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed11/progress.csv']

    pp = ['/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed12/progress.csv',
          '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed13/progress.csv',
          '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed14/progress.csv']

    pp = ['/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed15/progress.csv',
          '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed16/progress.csv',
          '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed17/progress.csv']

    pp = ['/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed18/progress.csv',
          '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed19/progress.csv',
          '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed20/progress.csv']

    pp = ['/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed23/progress.csv',
          '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed22/progress.csv',
          '/home/lthpc/Desktop/FOCAL-ICLR/output/sparse-point-robot/debug_seed21/progress.csv']

    oracle = data_read_2(
        paths=pp,key='Oracle_Prediction_Loss')
    maxi = data_read_2(
        paths=pp,key='Max_Prediction_Loss')
    mini = data_read_2(
        paths=pp,
        key='Min_Prediction_Loss')
    mean = data_read_2(
        paths=pp,
        key='Mean_Prediction_Loss')
    train = data_read_2(
        paths=pp,
        key='Prediction_Error_Train')
    accepted_return = data_read_2(
        paths=pp,
        key='Accpeted_Return')
    refused_return = data_read_2(
        paths=pp,
        key='Refused_Return')
    oracle_return = data_read_2(
        paths=pp,
        key='Oracle_Return')
    offline_return = data_read_2(
        paths=pp,
        key='AverageReturn_all_test_tasks')
    online_return = data_read_2(
        paths=pp,
        key='AverageReturn_all_test_tasks_online')
    na = data_read_2(
        paths=pp,
        key='Num_Accepted')

    #print(maml_data[0][-1],maml_data[1][-1])
    plt.figure()
    plt.xlabel('Million Environment Samples', size=20)
    plt.ylabel('Average Return', size=20)

    plot_full(oracle,'b','Oracle')
    plot_full(mini, 'r', 'Min')
    plot_full(mean, 'y', 'Mean')
    plot_full(maxi, 'purple', 'Max')
    plot_full(train, 'g', 'Train Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Million Environment Samples', size=20)
    plt.ylabel('Average Return', size=20)

    plot_full(accepted_return, 'b', 'Accepted')
    plot_full(refused_return, 'r', 'Refused')
    plot_full(oracle_return, 'g', 'Oracle')
    plot_full(na, 'y', 'Num_Accepted')

    #plt.plot(pearl_data[0], pearl_data[1], 'b')
    #plt.fill_between(pearl_data[0], pearl_data[1] - pearl_data[2], pearl_data[1] + pearl_data[2], color='b', alpha=0.2)
    #plt.plot(pearl_data[0], np.ones(pearl_data[0].shape) * np.max(pearl_data[1]), 'b:')
    plt.legend()

    
    plt.figure()
    plt.xlabel('Million Environment Samples', size=20)
    plt.ylabel('Average Return', size=20)

    plot_full(offline_return, 'b', 'FOCAL')
    plot_full(online_return, 'r', 'FOCAL Online')
    plt.legend()
    plt.show()
