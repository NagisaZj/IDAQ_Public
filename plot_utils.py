import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import copy
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import scipy
import scipy.stats as stats


def data_read_varibad(paths=[]):
	rewards = []
	if 'Point' in paths[0]:
		length = 330
	else:
		length = 130
	for p in paths:
		r = np.load(p)[:length]
		# r = smoothingaverage(r,20)
		rewards.append(r)

	mean = np.mean(rewards, 0)
	std = np.std(rewards, 0)
	# print(mean,std)
	xs = (np.arange(length) + 1) * 3200 * 25 / 1000000
	print(xs.shape, mean.shape, std.shape)
	confidence_interval = stats.bootstrap((xs,),np.mean,axis=1)
	low = confidence_interval.confidence_interval.low
	high = confidence_interval.confidence_interval.high
	mean = np.mean(low,high)
	interval_distance = high-mean
	# print(rewards,np.arange(rewards[0].shape[0]),rewards[0].shape[0])
	# print(xs,mean,std)

	return xs, mean, interval_distance, rewards


def data_read_varibad2(paths=[]):
	rewards = []
	length = 240
	for p in paths:
		r = np.load(p)[:length]
		# r = smoothingaverage(r,20)
		rewards.append(r)

	mean = np.mean(rewards, 0)
	std = np.std(rewards, 0)
	# print(mean,std)
	xs = (np.arange(length) + 1) * 15000 * 25 / 1000000
	print(xs.shape, mean.shape, std.shape)
	# print(rewards,np.arange(rewards[0].shape[0]),rewards[0].shape[0])
	# print(xs,mean,std)
	xs -= 0.19
	return xs, mean, std, rewards


def data_read(paths=['./outputfin2/cheetah-vel-sparse/2019_11_20_08_52_39/progress.csv',
                     '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_20_16_01_14/progress.csv',
                     '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_19_19_57_40/progress.csv'],
              load_name='AverageReturn_all_test_tasks'):
	mine_values = []
	num_trajs = len(paths)
	mine_paths = paths
	shortest = 10000000000
	for p in mine_paths:
		csv_data = pd.read_csv(p)
		values_steps = csv_data['Epoch'].values#*400*16*1000*2/1e6
		values_returns = csv_data[load_name].values
		# values_returns = smoothingaverage(values_returns)
		# print(values_steps.shape)
		length = values_returns.shape[0]
		shortest = length if length < shortest else shortest
		mine_values.append([values_steps, values_returns])
		# if 'pro-mp' in paths[0]:
		#     shortest = 1500
		'''plots = csv.reader(csvfile,delimiter=',')
		print(plots)
		for row in plots:
			print(row)'''
	'''if 'outputfin2' in paths[0]:
		shortest = shortest-10'''

	'''if 'rl2' in paths[0]:
		shortest = 700'''

	xs = mine_values[0][0][:shortest]
	ys = np.zeros([shortest, num_trajs])
	for i in range(num_trajs):
		ys[:, i] = mine_values[i][1][:shortest]
	mean = np.mean(ys, 1)
	std = np.std(ys, 1)
	# print(mean[-1], std[-1])
	# print(ys.shape)
	# ys[-20:,:] = np.mean(ys[-20:,:],0,keepdims=1)
	ys+=np.random.rand(ys.shape[0],ys.shape[1])*1e-3
	confidence_interval = stats.bootstrap((ys,),np.mean,axis=1)
	low = confidence_interval.confidence_interval.low
	high = confidence_interval.confidence_interval.high
	# print(low.shape,high.shape,ys.shape)
	mean = np.mean(np.array([low, high]),0)
	interval_distance = high-mean
	# print(rewards,np.arange(rewards[0].shape[0]),rewards[0].shape[0])
	# print(xs,mean,std)

	return xs, mean, interval_distance, ys.transpose()
	# return xs, mean, std, ys.transpose()


def data_read_npy(paths=['./outputfin2/cheetah-vel-sparse/2019_11_20_08_52_39/progress.csv',
                     '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_20_16_01_14/progress.csv',
                     '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_19_19_57_40/progress.csv'],
              load_name='AverageReturn_all_test_tasks'):
	mine_values = []
	num_trajs = len(paths)
	mine_paths = paths
	shortest = 10000000000
	for p in mine_paths:
		path = p + load_name+'.npy'
		data = np.load(path,allow_pickle=True)
		values_steps = np.arange(len(data))#*400*16*1000*2/1e6
		values_returns = data
		# values_returns = smoothingaverage(values_returns)
		# print(values_steps.shape)
		length = values_returns.shape[0]
		shortest = length if length < shortest else shortest
		mine_values.append([values_steps, values_returns])
		# if 'pro-mp' in paths[0]:
		#     shortest = 1500
		'''plots = csv.reader(csvfile,delimiter=',')
		print(plots)
		for row in plots:
			print(row)'''
	'''if 'outputfin2' in paths[0]:
		shortest = shortest-10'''

	'''if 'rl2' in paths[0]:
		shortest = 700'''

	xs = mine_values[0][0][:shortest]
	ys = np.zeros([shortest, num_trajs])
	for i in range(num_trajs):
		ys[:, i] = mine_values[i][1][:shortest]
	mean = np.mean(ys, 1)
	std = np.std(ys, 1)
	# print(mean[-1], std[-1])

	ys[-20:,:] = np.mean(ys[-20:,:],0,keepdims=1)
	ys+=np.random.rand(ys.shape[0],ys.shape[1])*1e-3
	confidence_interval = stats.bootstrap((ys,), np.mean, axis=1)
	low = confidence_interval.confidence_interval.low
	high = confidence_interval.confidence_interval.high
	# print(low.shape,high.shape,ys.shape)
	mean = np.mean(np.array([low, high]),0)
	# print(mean.shape)
	interval_distance = high - mean
	# print(mean.shape,interval_distance.shape)
	return xs, mean, interval_distance, ys.transpose()


def data_read_macaw(paths=['./outputfin2/cheetah-vel-sparse/2019_11_20_08_52_39/progress.csv',
                     '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_20_16_01_14/progress.csv',
                     '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_19_19_57_40/progress.csv']):
	mine_values = []
	num_trajs = len(paths)
	mine_paths = paths
	shortest = 10000000000
	for p in mine_paths:
		path = p +'/reward.npy'
		data = np.load(path,allow_pickle=True)
		values_steps = np.arange(len(data))*40*256*1000*2/1e6
		values_returns = data
		# values_returns = smoothingaverage(values_returns)
		# print(values_steps.shape)
		length = values_returns.shape[0]
		shortest = length if length < shortest else shortest
		mine_values.append([values_steps, values_returns])
		# if 'pro-mp' in paths[0]:
		#     shortest = 1500
		'''plots = csv.reader(csvfile,delimiter=',')
		print(plots)
		for row in plots:
			print(row)'''
	'''if 'outputfin2' in paths[0]:
		shortest = shortest-10'''

	'''if 'rl2' in paths[0]:
		shortest = 700'''

	xs = mine_values[0][0][:shortest]
	ys = np.zeros([shortest, num_trajs])
	for i in range(num_trajs):
		ys[:, i] = mine_values[i][1][:shortest]
	mean = np.mean(ys, 1)
	std = np.std(ys, 1)
	# print(mean[-1], std[-1])

	ys[-20:,:] = np.mean(ys[-20:,:],0,keepdims=1)
	ys+=np.random.rand(ys.shape[0],ys.shape[1])*1e-3
	confidence_interval = stats.bootstrap((ys,), np.mean, axis=1)
	low = confidence_interval.confidence_interval.low
	high = confidence_interval.confidence_interval.high
	# print(low.shape,high.shape,ys.shape)
	mean = np.mean(np.array([low, high]),0)
	interval_distance = high - mean
	return xs, mean, interval_distance, ys.transpose()


def data_read_success(paths=['./outputfin2/cheetah-vel-sparse/2019_11_20_08_52_39/progress.csv',
                             '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_20_16_01_14/progress.csv',
                             '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_19_19_57_40/progress.csv']):
	mine_values = []
	num_trajs = len(paths)
	mine_paths = paths
	shortest = 10000000000
	for p in mine_paths:
		csv_data = pd.read_csv(p)
		values_steps = csv_data['Number of env steps total'].values
		values_returns = csv_data['AverageSuccessRate_all_test_tasks'].values
		# values_returns = smoothingaverage(values_returns)
		# print(values_steps.shape)
		length = values_returns.shape[0]
		shortest = length if length < shortest else shortest
		mine_values.append([values_steps, values_returns])
		# if 'pro-mp' in paths[0]:
		#     shortest = 1500
		'''plots = csv.reader(csvfile,delimiter=',')
		print(plots)
		for row in plots:
			print(row)'''
	'''if 'outputfin2' in paths[0]:
		shortest = shortest-10'''

	'''if 'rl2' in paths[0]:
		shortest = 700'''

	xs = mine_values[0][0][:shortest] / 1e6
	ys = np.zeros([shortest, num_trajs])
	for i in range(num_trajs):
		ys[:, i] = mine_values[i][1][:shortest]
	mean = np.mean(ys, 1)
	std = np.std(ys, 1)
	print(mean[-1], std[-1])
	return xs, mean, std, ys.transpose()


def data_read_success2(paths=['./outputfin2/cheetah-vel-sparse/2019_11_20_08_52_39/progress.csv',
                              '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_20_16_01_14/progress.csv',
                              '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_19_19_57_40/progress.csv']):
	mine_values = []
	num_trajs = len(paths)
	mine_paths = paths
	shortest = 10000000000
	for p in mine_paths:
		csv_data = pd.read_csv(p)
		values_steps = csv_data['Number of env steps total'].values
		values_returns = csv_data['AverageSuccess_all_test_tasks_last'].values
		# values_returns = smoothingaverage(values_returns)
		# print(values_steps.shape)
		length = values_returns.shape[0]
		shortest = length if length < shortest else shortest
		mine_values.append([values_steps, values_returns])
		# if 'pro-mp' in paths[0]:
		#     shortest = 1500
		'''plots = csv.reader(csvfile,delimiter=',')
		print(plots)
		for row in plots:
			print(row)'''
	'''if 'outputfin2' in paths[0]:
		shortest = shortest-10'''

	'''if 'rl2' in paths[0]:
		shortest = 700'''

	xs = mine_values[0][0][:shortest] / 1e6
	ys = np.zeros([shortest, num_trajs])
	for i in range(num_trajs):
		ys[:, i] = mine_values[i][1][:shortest]
	mean = np.mean(ys, 1)
	std = np.std(ys, 1)
	print(mean[-1], std[-1])
	return xs, mean, std, ys.transpose()


def data_read_Maesn(paths=['./outputfin2/cheetah-vel-sparse/2019_11_20_08_52_39/progress.csv',
                           '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_20_16_01_14/progress.csv',
                           '/home/zj/Desktop/new-pearl/outputfin2/cheetah-vel-sparse/2019_11_19_19_57_40/progress.csv']):
	mine_values = []
	num_trajs = len(paths)
	mine_paths = paths
	shortest = 10000000000
	for p in mine_paths:
		csv_data = pd.read_csv(p, engine='python')
		values_returns = csv_data['1AverageReturn'].values
		# values_returns = smoothingaverage(values_returns)
		# print(values_steps.shape)
		length = values_returns.shape[0]
		if 'Point' or 'Reach' in p:
			val = 3
		else:
			val = 1
		values_steps = np.arange(length) * 6400 * val
		shortest = length if length < shortest else shortest
		mine_values.append([values_steps, values_returns])
		'''plots = csv.reader(csvfile,delimiter=',')
		print(plots)
		for row in plots:
			print(row)'''

	xs = mine_values[0][0][:shortest] / 1e6
	ys = np.zeros([shortest, num_trajs])
	for i in range(num_trajs):
		ys[:, i] = mine_values[i][1][:shortest]
	mean = np.mean(ys, 1)
	std = np.std(ys, 1)
	print(mean[-1], std[-1])
	return xs, mean, std, ys.transpose()


def data_read_mame(paths=[]):
	rewards = []
	length = 400
	for p in paths:
		r = np.load(p + '/reward.npy', allow_pickle=True)[:100]
		x = np.load(p + '/step.npy', allow_pickle=True)[:100]
		if 'metaworld' in p:
			r = np.load(p + '/success.npy', allow_pickle=True)[:100]
		print(x[0].data.cpu().numpy().astype(int))
		rewards.append(r)
		print(x, r)

	mean = np.mean(rewards, 0)
	std = np.std(rewards, 0)
	# print(mean,std)
	stepsize = x[0].data.cpu().numpy().astype(int) / rewards[0].shape[0]
	xs = (np.arange(100)) * stepsize / 1e6
	print(xs.shape, mean.shape, std.shape)
	# print(rewards,np.arange(rewards[0].shape[0]),rewards[0].shape[0])
	# print(xs,mean,std)
	return xs, mean, std, rewards


def plot_full(data, color, name):
	plt.plot(data[0], data[1], color, label=name)
	plt.fill_between(data[0], data[1] - data[2], data[1] + data[2], color=color, alpha=0.3, linewidth=0)
	plt.plot(data[0], np.ones(data[0].shape) * np.mean(data[1][-40:]), color=color, linestyle='--')


def smoothingaverage(data, window_size=30):
	window = np.ones(int(window_size)) / float(window_size)
	return np.convolve(data, window, 'same')


def smooth(data, smooth_range):
	print('hhhhhhh', type(data), len(data))
	new_data = np.zeros_like(data)
	for i in range(0, data.shape[-1]):
		if i < smooth_range:
			new_data[:, i] = 1. * np.sum(data[:, :i + 1], axis=1) / (i + 1)
		else:
			new_data[:, i] = 1. * np.sum(data[:, i - smooth_range + 1:i + 1], axis=1) / smooth_range

	return new_data


def new_plot_full_xuxian(data_full, color, name):
	data = data_full[-1]
	length = len(data[0])
	index = data_full[0]
	data = np.minimum(data, 10e30)
	data = smooth(data, config['smooth_range'])
	# data[:, 0] = 1e-30
	data_std = np.std(data, axis=0) * config['data_scale']
	data_mean = np.mean(data, axis=0) * config['data_scale']
	data_median = np.median(data, axis=0) * config['data_scale']
	data_sort = np.sort(data, axis=0) * config['data_scale']
	data_min = data_sort[config['get_min'], :]
	data_max = data_sort[config['get_max'], :]
	ax.plot(index[: len(data[0])], np.ones(len(data[0])) * np.mean(data_mean[-40:]), color=color, linestyle='--',
	        linewidth=2.0)


# ax.fill_between(index[: len(data[0])], data_min, data_max, alpha=0.1, color=color,
#                linewidth=0)

# ax.plot(index[: len(data[0])], data_median, color=color,
#        label=name, linewidth=config['linewidth'])

def new_plot_full(data_full, color, name):
	data = data_full[-1]
	length = len(data[0])
	index = data_full[0]
	data = np.minimum(data, 10e30)
	data = smooth(data, config['smooth_range'])
	# data[:, 0] = 1e-30
	data_std = np.std(data, axis=0) * config['data_scale']
	data_mean = np.mean(data, axis=0) * config['data_scale']
	data_median = np.median(data, axis=0) * config['data_scale']
	data_sort = np.sort(data, axis=0) * config['data_scale']
	data_min = data_sort[config['get_min'], :]
	data_max = data_sort[config['get_max'], :]
	print('min', np.min(data))
	ax.fill_between(index[: len(data[0])], data_mean - data_std, data_mean + data_std, alpha=0.08, color=color,
	                linewidth=0)
	ax.plot(index[: len(data[0])], data_mean, '--' if 'Expert' in name else '', color=color,
	        label=name, linewidth=config['linewidth'],)
	# ax.fill_between(index[: len(data[0])], data_min, data_max, alpha=0.1, color=color,
	#                linewidth=0)

	# ax.plot(index[: len(data[0])], data_median, color=color,
	#        label=name, linewidth=config['linewidth'])

	print(len(data[0]), index[-2])


color_set = {
	'Amaranth': np.array([0.9, 0.17, 0.31]),  # main algo
	'Amber': np.array([1.0, 0.49, 0.0]),  # main baseline
	'Bleu de France': np.array([0.19, 0.55, 0.91]),
	'Electric violet': np.array([0.56, 0.0, 1.0]),
	'Dark sea green': 'forestgreen',
	'Dark electric blue': 'brown',
	'Dark gray': np.array([0.66, 0.66, 0.66]),
	'Arsenic': np.array([0.23, 0.27, 0.29]),
	'Novel': 'steelblue',
}

color_list = []
for key, value in color_set.items():
	color_list.append(value)

plot_config_default = {
	'legend_loc': 'best',
	'legend_ncol': 1,
	'legend_prop_size': 24.0,
	'legend_prefix': '',
	'data_scale': 1,
	'linewidth': 3,
	'smooth_range': 5,
	'framealpha': 0.6,
	'get_min': 0,
	'get_max': -1,
}

plt_config_point = {
	'data_scale': 1,
	'legend_loc': 'best',
	'legend_ncol': 1,
	'legend_prop_size': 19.0,
	'xlabel': 'Iterations',
	'ylabel': 'Average Return',
	'xlim': (-5, 55),
	'ylim': (-1, 13),
	'color': {
		'QPLEX': color_set['Amaranth'],
		'QTRAN': color_set['Amber'],
		'QMIX': color_set['Electric violet'],
		'Qatten': color_set['Bleu de France'],
		'VDN': color_set['Dark sea green'],
		'IQL': color_set['Dark gray'],
	},
	'smooth_range': 5,
	'framealpha': 1,
	'get_min': 0,
	'get_max': -1
}

plt_config_cheetah = {
	'data_scale': 1,
	'legend_loc': 'best',
	'legend_ncol': 1,
	'legend_prop_size': 18.0,
	'xlabel': 'Million Environment Samples',
	'ylabel': 'Average Return',
	'xlim': (-0.2, 4.5),
	'ylim': (-5, 115),
	'color': {
		'QPLEX': color_set['Amaranth'],
		'QTRAN': color_set['Amber'],
		'QMIX': color_set['Electric violet'],
		'Qatten': color_set['Bleu de France'],
		'VDN': color_set['Dark sea green'],
		'IQL': color_set['Dark gray'],
	},
	'smooth_range': 20,
	'framealpha': 1,
	'get_min': 0,
	'get_max': -1
}

plt_config_walker = {
	'data_scale': 1,
	'legend_loc': 'best',
	'legend_ncol': 1,
	'legend_prop_size': 8.0,
	'xlabel': 'Million Environment Samples',
	'ylabel': 'Average Return',
	'xlim': (-0.2, 2.7),
	'ylim': (-20, 100),
	'color': {
		'QPLEX': color_set['Amaranth'],
		'QTRAN': color_set['Amber'],
		'QMIX': color_set['Electric violet'],
		'Qatten': color_set['Bleu de France'],
		'VDN': color_set['Dark sea green'],
		'IQL': color_set['Dark gray'],
	},
	'smooth_range': 20,
	'framealpha': 1,
	'get_min': 0,
	'get_max': -1
}

plt_config_reacher = {
	'data_scale': 1,
	'legend_loc': 'best',
	'legend_ncol': 1,
	'legend_prop_size': 10.0,
	'xlabel': 'Million Environment Samples',
	'ylabel': 'Average Return',
	'xlim': (-0.2, 3.4),
	'ylim': (-1.5, 8),
	'color': {
		'QPLEX': color_set['Amaranth'],
		'QTRAN': color_set['Amber'],
		'QMIX': color_set['Electric violet'],
		'Qatten': color_set['Bleu de France'],
		'VDN': color_set['Dark sea green'],
		'IQL': color_set['Dark gray'],
	},
	'smooth_range': 20,
	'framealpha': 1,
	'get_min': 0,
	'get_max': -1
}

plt_config_params = {
	'data_scale': 1,
	'legend_loc': 'best',
	'legend_ncol': 1,
	'legend_prop_size': 8.0,
	'xlabel': 'Million Environment Samples',
	'ylabel': 'Average Return',
	'xlim': (-0.2, 4.8),
	'ylim': (-1.5, 40),
	'color': {
		'QPLEX': color_set['Amaranth'],
		'QTRAN': color_set['Amber'],
		'QMIX': color_set['Electric violet'],
		'Qatten': color_set['Bleu de France'],
		'VDN': color_set['Dark sea green'],
		'IQL': color_set['Dark gray'],
	},
	'smooth_range': 20,
	'framealpha': 1,
	'get_min': 0,
	'get_max': -1
}
plt_config_meta = {
	'data_scale': 1,
	'legend_loc': 'best',
	'legend_ncol': 1,
	'legend_prop_size': 8.0,
	'xlabel': 'Million Environment Samples',
	'ylabel': 'Average Success Rate',
	'xlim': (-0.2, 5.34),
	'ylim': (-0.05, 0.37),
	'color': {
		'QPLEX': color_set['Amaranth'],
		'QTRAN': color_set['Amber'],
		'QMIX': color_set['Electric violet'],
		'Qatten': color_set['Bleu de France'],
		'VDN': color_set['Dark sea green'],
		'IQL': color_set['Dark gray'],
	},
	'smooth_range': 20,
	'framealpha': 1,
	'get_min': 0,
	'get_max': -1
}

plt_config_meta_2 = {
	'data_scale': 1,
	'legend_loc': 'best',
	'legend_ncol': 1,
	'legend_prop_size': 8.0,
	'xlabel': 'Million Environment Samples',
	'ylabel': 'Average Success Rate',
	'xlim': (-0.2, 11.36),
	'ylim': (-0.05, 0.46),
	'color': {
		'QPLEX': color_set['Amaranth'],
		'QTRAN': color_set['Amber'],
		'QMIX': color_set['Electric violet'],
		'Qatten': color_set['Bleu de France'],
		'VDN': color_set['Dark sea green'],
		'IQL': color_set['Dark gray'],
	},
	'smooth_range': 20,
	'framealpha': 1,
	'get_min': 0,
	'get_max': -1
}


def smooth(data, smooth_range):
	print('hhhhhhh', type(data), len(data))
	new_data = np.zeros_like(data)
	for i in range(0, data.shape[-1]):
		if i < smooth_range:
			new_data[:, i] = 1. * np.sum(data[:, :i + 1], axis=1) / (i + 1)
		else:
			new_data[:, i] = 1. * np.sum(data[:, i - smooth_range + 1:i + 1], axis=1) / smooth_range

	return new_data


def config_reduce(config_primal, key_reduce):
	config_copy = copy.deepcopy(config_primal)
	for key in config_copy.keys():
		if type(config_copy[key]) == dict:
			if key_reduce in config_copy[key].keys():
				config_copy[key] = config_copy[key][key_reduce]
	return copy.deepcopy(config_copy)


def config_set_default(config_primal):
	config_copy = copy.deepcopy(config_primal)
	for key in plot_config_default.keys():
		if not (key in config_copy.keys()):
			config_copy[key] = plot_config_default[key]
	return copy.deepcopy(config_copy)


config = plt_config_point
# config = plt_config_cheetah
# config = plt_config_walker
# config = plt_config_reacher
# config = plt_config_params
# config = plt_config_meta
# config = plt_config_meta_2
# config = config_reduce(config, args.id[2:])
config = config_set_default(config)
if 'figlegend' in config.keys():
	figure = plt.figure(figsize=(config['figlegend'], 4.8))
else:
	figure = plt.figure(figsize=(8.5, 6))
# figure = plt.figure(figsize=(6.5,8))
plt.style.use('seaborn-whitegrid')
plt.rc('font', family='Times New Roman')
# matplotlib.rcParams['text.usetex'] = True
plt.clf()
ax = plt.gca()


def plot_all(datas, legends, start=0):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_color('black')
	ax.spines['bottom'].set_color('black')

	# plt.xlim(config['xlim'])
	# plt.ylim(config['ylim'])
	plt.tick_params('x', labelsize=20.0)
	plt.tick_params('y', labelsize=20.0)
	plt.xlabel('Iterations', {'size': 26.0})
	plt.ylabel(config['ylabel'], {'size': 26.0})
	ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
	ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
	if config['xlabel'] == 'Epoches':
		def formatnum_epoch(x, pos):
			return int(x // (5e3))

		formatter = FuncFormatter(formatnum_epoch)
		ax.xaxis.set_major_formatter(formatter)
	elif config['ylabel'] == 'Median Test Return':
		def formatnum(x, pos):
			return str(x / 1e3) + 'K'  # '$%.1f$M' % (x / 1e6)

		formatter = FuncFormatter(formatnum)
		ax.xaxis.set_major_formatter(formatter)
	elif config['xlim'][1] > 1e5:
		def formatnum(x, pos):
			return str(x / 1e6) + 'M'  # '$%.1f$M' % (x / 1e6)

		formatter = FuncFormatter(formatnum)
		ax.xaxis.set_major_formatter(formatter)
	if config['ylim'][1] > 1e5:
		plt.yscale('log')

	# for i in range(len(datas)-1,0,-1):
	#    new_plot_full_xuxian(datas[i], color_list[i+start], legends[i])
	new_plot_full(datas[0], color_list[0], legends[0])
	# for i in range(1,len(datas)):
	# for i in range(len(datas) - 1, 0, -1):
	for i in range(1,len(datas)):
		new_plot_full(datas[i], color_list[i + start], legends[i])
	# new_plot_full_xuxian(datas[0], color_list[0], legends[0])
	# new_plot_full(datas[0], color_list[0], legends[0])

	plt.legend(loc=config['legend_loc'], prop={'size': config['legend_prop_size']}, frameon=True,
	           framealpha=config['framealpha'], facecolor='white', ncol=config['legend_ncol'])


def legend():
	if 'figlegend' in config.keys():
		plt.figlegend(loc='upper right', prop={'size': 26.0}, frameon=True, ncol=1)
		plt.tight_layout(rect=(0, 0, 6.4 / config['figlegend'], 1))
	else:
		if not (config['legend_loc'] is None):
			plt.legend(loc=config['legend_loc'], prop={'size': config['legend_prop_size']}, frameon=True,
			           framealpha=config['framealpha'], facecolor='white', ncol=config['legend_ncol'])


