import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import click

from plot_utils import *

@click.command()
@click.option('--name', default='cheetah-vel__2022-08-25_09-17-39')

def main(name):


	names = ['push-v2__2022-08-30_13-05-21','push-v2__2022-08-30_13-05-31','push-v2__2022-08-30_13-05-41','push-v2__2022-08-30_13-05-52']
	names = ['pick-place-v2__2022-08-30_10-07-37', 'pick-place-v2__2022-08-30_10-07-44', 'pick-place-v2__2022-08-30_10-07-45',
			 'pick-place-v2__2022-08-30_10-07-52']
	paths = ['output/' + name + '/pick-place-v2/debug/progress.csv' for name in names]
	# path = 'output/' + name + '/reach-v2/debug/'
	mine_testing_data = \
		data_read(paths=paths,
		          load_name='AverageReturn_all_test_tasks')

	mine_training_context_data = \
		data_read(paths,
		          load_name='AverageTrainReturn_all_train_tasks')

	mine_training_data = \
		data_read(paths,
		          load_name='AverageReturn_all_train_tasks')

	# print(maml_data[0][-1],maml_data[1][-1])
	datas = [mine_testing_data, mine_training_context_data, mine_training_data]
	legends = ['CPEARL-testing', 'CPEARL-training-context', 'CPEARL-training']
	# datas = [mine_data_new_intr, promp_data, erl2_data, mame_data]
	# legends = ['MetaCURE', 'ProMP', 'E-RL^2', 'MAME']
	plot_all(datas, legends, 0)
	plt.title('Push-V2', size=30)
	# plt.plot(mine_data_new_intr[0], np.ones(mine_data_new_intr[0].shape) * 9.98, color='olive', linestyle='--',
	#         linewidth=2, label='EPI')
	# legend()
	plt.tight_layout()
	plt.show()
	# plt.savefig("figures/curves/" + name + ".png")

if __name__ =="__main__":
	main()
