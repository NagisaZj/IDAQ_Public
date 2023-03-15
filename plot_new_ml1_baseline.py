import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import click

from plot_utils import *

@click.command()
@click.option("--name", default="Sparse-Point-Robot")
def main(name):
	exp_dict = {

		"Sweep-V2Med": ["sweep-v2__2023-01-25_10-25-43", "sweep-v2__2023-01-25_10-25-44"],
		"Peg-Insert-Side-V2Med": ["peg-insert-side-v2__2023-01-25_10-34-42", "peg-insert-side-v2__2023-01-25_10-35-01"],
		"Push-V2": ["push-v2__2023-01-17_20-09-13", "push-v2__2023-01-17_20-09-27", "push-v2__2023-01-19_12-01-31", "push-v2__2023-01-19_12-01-45", "push-v2__2023-01-19_12-02-01", "push-v2__2023-01-19_12-02-15"],
		# ["push-v2__2022-09-01_09-57-19", "push-v2__2022-09-01_09-57-29", ],
		"Reach-V2": ["reach-v2__2023-01-18_11-25-20", "reach-v2__2023-01-18_11-25-34"],
		"Pick-Place-V2": ["pick-place-v2__2023-01-17_20-09-54", "pick-place-v2__2023-01-17_20-10-09","pick-place-v2__2023-01-21_14-37-58","pick-place-v2__2023-01-21_14-38-13"],
		"Drawer-Close-V2": ["drawer-close-v2__2023-01-17_20-45-05", "drawer-close-v2__2023-01-17_20-45-06"],
		"Sparse-Point-Robot": ["sparse-point-robot__2023-01-17_17-20-32", "sparse-point-robot__2023-01-17_17-20-33"],
		"Soccer-V2": ["soccer-v2__2023-01-18_11-26-45", "soccer-v2__2023-01-18_11-26-46"], ###

	}



	# drawer-close-v2  peg-insert-side-v2 pick-place-v2 push-v2
	name = name
	ps = exp_dict[name]
	ori_name = name
	name = name if "Medium" not in name else  name[:-7]
	if "Medium" not in name and "Med" in name:
		name = name[:-3]


	# print(name,ps)
	paths = ["output/" + n + "/"+name.lower()+"/debug/progress.csv" for n in ps]

	mine_testing_data = \
		data_read(paths=paths,
				  load_name="AverageReturn_all_test_tasks")

	std_mine_testing_data = \
		data_read(paths=paths,
				  load_name="AverageReturn_std_all_test_tasks")

	datas = [mine_testing_data,std_mine_testing_data]

	print(ori_name)
	means = []
	distances = []


	returns = []
	path = './data/'+name.lower()
	for i in range(50):
		for j in range(45):
			file_name = path+'/goal_idx%d'%i+'/trj_evalsample%d_step49500.npy'%j
			traj = np.load(file_name,allow_pickle=1)
			returns.append(sum(s[2] for s in traj))


	average_return = np.mean(returns)

	# average_return = 1


	for i in range(len(datas)):
		mean = np.mean(datas[i][1][-20:])
		interval_distance = np.mean(datas[i][2][-20:])
		means.append(mean)
		distances.append(interval_distance)
	print(['%.2f$~\pm~$%.2f & '%(m/average_return,d/average_return) for (m,d) in zip(means,distances)])
	print("%.2f"%average_return)

	# datas = [mine_data_new_intr, promp_data, erl2_data, mame_data]
	# legends = ["MetaCURE", "ProMP", "E-RL^2", "MAME"]
	# plot_all(datas, legends, 0)
	# plt.title(ori_name, size=30)
	# # plt.plot(mine_data_new_intr[0], np.ones(mine_data_new_intr[0].shape) * 9.98, color="olive", linestyle="--",
	# #         linewidth=2, label="EPI")
	# # legend()
	# plt.tight_layout()
	# plt.show()
	# # plt.savefig("./figures/" + ori_name + ".png")
	# plt.close()
	return


	names = ["push-v2__2022-08-31_16-11-45","push-v2__2022-08-31_16-11-55"]
	names = ["pick-place-v2__2022-08-31_16-10-59", "pick-place-v2__2022-08-31_16-11-08", "pick-place-v2__2022-08-31_16-10-59", "pick-place-v2__2022-08-31_16-11-08",]
	# names = ["reach-v2__2022-08-31_16-11-23", "reach-v2__2022-08-31_16-11-33",]
	# names = ["reach-v2__2022-08-31_18-57-30", "reach-v2__2022-08-31_18-57-40",]
	names = ["window-open-v2__2022-08-31_16-12-07", "window-open-v2__2022-08-31_16-12-15", ]
	names = ["drawer-close-v2__2022-08-31_16-12-25", "drawer-close-v2__2022-08-31_16-12-34", ]
	# names = ["drawer-close-v2__2022-08-31_16-12-34", "drawer-close-v2__2022-08-31_16-12-34", ]



	names = ["reach-v2__2022-09-01_09-57-07","reach-v2__2022-09-01_09-57-09"]
	names = ["reach-v2__2022-09-01_11-10-02", "reach-v2__2022-09-01_11-10-12"]
	names = ["reach-v2__2022-09-01_11-16-47", "reach-v2__2022-09-01_11-16-58","reach-v2__2022-09-01_18-53-27", "reach-v2__2022-09-01_18-53-35"] #good
	# names = ["reach-v2__2022-09-01_14-13-01", "reach-v2__2022-09-01_14-13-10"]
	# names = ["reach-v2__2022-09-01_18-53-27", "reach-v2__2022-09-01_18-53-35"]
	# names = ["reach-v2__2022-09-01_18-53-46", "reach-v2__2022-09-01_18-53-55"]
	names = ["drawer-close-v2__2022-09-01_09-58-21", "drawer-close-v2__2022-09-01_09-58-31", ] #good
	# names = ["window-open-v2__2022-09-01_09-58-02", "window-open-v2__2022-09-01_09-58-11", ]
	names = ["pick-place-v2__2022-09-01_09-57-39", "pick-place-v2__2022-09-01_09-57-51", ]
	names = ["push-v2__2022-09-01_09-57-19", "push-v2__2022-09-01_09-57-29", ]
	# names = ["pick-place-v2__2022-09-01_21-41-30", "pick-place-v2__2022-09-01_21-41-39", ]
	names = ["peg-insert-side-v2__2022-09-02_10-12-28", "peg-insert-side-v2__2022-09-02_10-12-37", "peg-insert-side-v2__2022-09-02_10-13-04","peg-insert-side-v2__2022-09-02_10-13-14"]
	names = ["lever-pull-v2__2022-09-02_10-11-21", "lever-pull-v2__2022-09-02_10-11-29", "lever-pull-v2__2022-09-02_10-11-48","lever-pull-v2__2022-09-02_10-12-00"]



	names = ["peg-insert-side-v2__2022-09-03_09-39-22", "peg-insert-side-v2__2022-09-03_09-39-32",
			 "peg-insert-side-v2__2022-09-03_09-39-41", "peg-insert-side-v2__2022-09-03_09-39-49"]
	# names = ["window-open-v2__2022-09-03_09-42-21", "window-open-v2__2022-09-03_09-42-29",
	# 		 "window-open-v2__2022-09-03_09-42-40", "window-open-v2__2022-09-03_09-42-48"]  # good
	# names = ["handle-pull-v2__2022-09-03_09-44-05", "handle-pull-v2__2022-09-03_09-44-14",
	# 		 "handle-pull-v2__2022-09-03_09-44-25", "handle-pull-v2__2022-09-03_09-44-36"] #bad
	# names = ["handle-pull-side-v2__2022-09-03_09-47-09", "handle-pull-side-v2__2022-09-03_09-47-18",
	# 		 "handle-pull-side-v2__2022-09-03_09-47-28", ]#bad
	names = ["handle-pull-v2__2022-09-03_18-23-49", "handle-pull-v2__2022-09-03_18-23-59",
			 "handle-pull-v2__2022-09-03_18-24-10", "handle-pull-v2__2022-09-03_18-24-20"] # good
	# names = ["handle-pull-side-v2__2022-09-03_18-22-54", "handle-pull-side-v2__2022-09-03_18-23-03",
	# 		 "handle-pull-side-v2__2022-09-03_18-23-13", ]
	# names = ["pick-out-of-hole-v2__2022-09-03_20-43-42", "pick-out-of-hole-v2__2022-09-03_20-43-53",
	# 		 "pick-out-of-hole-v2__2022-09-03_20-44-03", "pick-out-of-hole-v2__2022-09-03_20-44-12"]

	# names = ["plate-slide-side-v2__2022-09-04_09-37-31", "plate-slide-side-v2__2022-09-04_09-37-10",
	# 		 "plate-slide-side-v2__2022-09-04_09-37-21", "plate-slide-side-v2__2022-09-04_09-37-31"] # good
	# names = ["plate-slide-v2__2022-09-04_09-40-20", "plate-slide-v2__2022-09-04_09-40-30",
	# 		 "plate-slide-v2__2022-09-04_09-40-40", "plate-slide-v2__2022-09-04_09-40-49"] # good
	# names = ["reach-wall-v2__2022-09-04_22-22-01", "reach-wall-v2__2022-09-04_22-22-12",
	# 		  		 "reach-wall-v2__2022-09-04_22-22-21","reach-wall-v2__2022-09-04_22-22-32",] # good

	# names = ["soccer-v2__2022-09-04_09-37-55", "soccer-v2__2022-09-04_09-38-04",
 	# 	 "soccer-v2__2022-09-04_09-38-14",]

	# names = ["push-wall-v2__2022-09-04_22-20-58", "push-wall-v2__2022-09-04_22-21-08",
	# 		  		 "push-wall-v2__2022-09-04_22-21-19","push-wall-v2__2022-09-04_22-21-28",] # good



	names = ["sweep-v2__2022-09-05_09-00-46", "sweep-v2__2022-09-05_09-00-55",
			  		 "sweep-v2__2022-09-05_09-01-06","sweep-v2__2022-09-05_09-01-15",] # good
	# names = ["shelf-place-v2__2022-09-05_08-59-34", "shelf-place-v2__2022-09-05_08-59-44",
	# 		  		 "shelf-place-v2__2022-09-05_08-59-54","shelf-place-v2__2022-09-05_09-00-04",]# good
	# names = ["window-close-v2__2022-09-05_21-08-16", "window-close-v2__2022-09-05_21-08-26",
	# 		  		 "window-close-v2__2022-09-05_21-08-35",] # good



	###medium
	names = ["reach-v2__2022-09-06_19-33-09", "reach-v2__2022-09-06_19-33-18",
			 "reach-v2__2022-09-06_19-33-28", "reach-v2__2022-09-06_19-33-36", ]
	# names = ["handle-pull-v2__2022-09-06_19-31-14", "handle-pull-v2__2022-09-06_19-31-23",
	# 		 "handle-pull-v2__2022-09-06_19-31-28", "handle-pull-v2__2022-09-06_19-31-33", ]
	# names = ["push-wall-v2__2022-09-06_19-31-48", "push-wall-v2__2022-09-06_19-31-59",
	# 		 "push-wall-v2__2022-09-06_19-32-09", "push-wall-v2__2022-09-06_19-32-20", ]
	# names = ["window-open-v2__2022-09-06_19-34-38", "window-open-v2__2022-09-06_19-34-47",
	# 		 "window-open-v2__2022-09-06_19-34-54",  ] # good

	names = ["peg-insert-side-v2__2022-09-03_09-39-22", "peg-insert-side-v2__2022-09-03_09-39-32",
			 "peg-insert-side-v2__2022-09-03_09-39-41", "peg-insert-side-v2__2022-09-03_09-39-49"]
	names = ["peg-insert-side-v2__2022-09-02_10-12-28", "peg-insert-side-v2__2022-09-02_10-12-37", "peg-insert-side-v2__2022-09-02_10-13-04","peg-insert-side-v2__2022-09-02_10-13-14"]

	paths = ["output/" + name + "/peg-insert-side-v2/debug/progress.csv" for name in names]
	# path = "output/" + name + "/reach-v2/debug/"
	mine_testing_data = \
		data_read(paths=paths,
		          load_name="AverageReturn_all_test_tasks")

	mine_training_context_data = \
		data_read(paths,
		          load_name="AverageTrainReturn_all_train_tasks")

	mine_training_data = \
		data_read(paths,
		          load_name="AverageReturn_all_train_tasks")

	# print(maml_data[0][-1],maml_data[1][-1])
	datas = [mine_testing_data, mine_training_context_data, mine_training_data]
	legends = ["CPEARL-testing", "CPEARL-training-context", "CPEARL-training"]
	# datas = [mine_data_new_intr, promp_data, erl2_data, mame_data]
	# legends = ["MetaCURE", "ProMP", "E-RL^2", "MAME"]
	plot_all(datas, legends, 0)
	plt.title("Reach-V2", size=30)
	# plt.plot(mine_data_new_intr[0], np.ones(mine_data_new_intr[0].shape) * 9.98, color="olive", linestyle="--",
	#         linewidth=2, label="EPI")
	# legend()
	plt.tight_layout()
	plt.show()
	# plt.savefig("figures/curves/" + name + ".png")

if __name__ =="__main__":
	main()
	# exp_dict = {
	# 	"Push-V2": ["push-v2__2022-09-01_09-57-19", "push-v2__2022-09-01_09-57-29", ],
	# 	"Reach-V2": ["reach-v2__2022-09-01_11-16-47", "reach-v2__2022-09-01_11-16-58", "reach-v2__2022-09-01_18-53-27",
	# 				 "reach-v2__2022-09-01_18-53-35"],
	# 	"Pick-Place-V2": ["pick-place-v2__2022-09-01_09-57-39", "pick-place-v2__2022-09-01_09-57-51", ],
	# 	"Peg-Insert-Side-V2": ["peg-insert-side-v2__2022-09-02_10-12-28", "peg-insert-side-v2__2022-09-02_10-12-37",
	# 						   "peg-insert-side-v2__2022-09-02_10-13-04", "peg-insert-side-v2__2022-09-02_10-13-14"],
	# 	"Window-Open-V2": ["window-open-v2__2022-09-03_09-42-21", "window-open-v2__2022-09-03_09-42-29",
	# 					   "window-open-v2__2022-09-03_09-42-40", "window-open-v2__2022-09-03_09-42-48"],
	# 	"Drawer-Close-V2": ["drawer-close-v2__2022-09-01_09-58-21", "drawer-close-v2__2022-09-01_09-58-31", ],
	# 	"Lever-Pull-V2": ["lever-pull-v2__2022-09-02_10-11-21", "lever-pull-v2__2022-09-02_10-11-29",
	# 					  "lever-pull-v2__2022-09-02_10-11-48", "lever-pull-v2__2022-09-02_10-12-00"],
	# 	"Handle-Pull-V2": ["handle-pull-v2__2022-09-03_18-23-49", "handle-pull-v2__2022-09-03_18-23-59",
	# 					   "handle-pull-v2__2022-09-03_18-24-10", "handle-pull-v2__2022-09-03_18-24-20"],
	# 	"Handle-Pull-Side-V2": ["handle-pull-side-v2__2022-09-03_18-22-54", "handle-pull-side-v2__2022-09-03_18-23-03",
	# 							"handle-pull-side-v2__2022-09-03_18-23-13", ],
	# 	"Pick-Out-Of-Hole-V2": ["pick-out-of-hole-v2__2022-09-03_20-43-42", "pick-out-of-hole-v2__2022-09-03_20-43-53",
	# 							"pick-out-of-hole-v2__2022-09-03_20-44-03", "pick-out-of-hole-v2__2022-09-03_20-44-12"],
	# 	"Plate-Slide-Side-V2": ["plate-slide-side-v2__2022-09-04_09-37-31", "plate-slide-side-v2__2022-09-04_09-37-10",
	# 							"plate-slide-side-v2__2022-09-04_09-37-21", "plate-slide-side-v2__2022-09-04_09-37-31"],
	# 	"Plate-Slide-V2": ["plate-slide-v2__2022-09-04_09-40-20", "plate-slide-v2__2022-09-04_09-40-30",
	# 					   "plate-slide-v2__2022-09-04_09-40-40", "plate-slide-v2__2022-09-04_09-40-49"],
	# 	"Reach-Wall-V2": ["reach-wall-v2__2022-09-04_22-22-01", "reach-wall-v2__2022-09-04_22-22-12",
	# 					  "reach-wall-v2__2022-09-04_22-22-21", "reach-wall-v2__2022-09-04_22-22-32", ],
	# 	"Soccer-V2": ["soccer-v2__2022-09-04_09-37-55", "soccer-v2__2022-09-04_09-38-04",
	# 				  "soccer-v2__2022-09-04_09-38-14", ],
	# 	"Push-Wall-V2": ["push-wall-v2__2022-09-04_22-20-58", "push-wall-v2__2022-09-04_22-21-08",
	# 					 "push-wall-v2__2022-09-04_22-21-19", "push-wall-v2__2022-09-04_22-21-28", ],
	# 	"Window-Close-V2": ["window-close-v2__2022-09-05_21-08-16", "window-close-v2__2022-09-05_21-08-26",
	# 						"window-close-v2__2022-09-05_21-08-35", ],
	# 	"Shelf-Place-V2": ["shelf-place-v2__2022-09-05_08-59-34", "shelf-place-v2__2022-09-05_08-59-44",
	# 					   "shelf-place-v2__2022-09-05_08-59-54", "shelf-place-v2__2022-09-05_09-00-04", ],
	# 	"Sweep-V2": ["sweep-v2__2022-09-05_09-00-46", "sweep-v2__2022-09-05_09-00-55", "sweep-v2__2022-09-05_09-01-06",
	# 				 "sweep-v2__2022-09-05_09-01-15", ],
	# 	"Reach-V2 Medium": ["reach-v2__2022-09-06_19-33-09", "reach-v2__2022-09-06_19-33-18",
	# 						"reach-v2__2022-09-06_19-33-28", "reach-v2__2022-09-06_19-33-36", ],
	# 	"Handle-Pull-V2 Medium": [["handle-pull-v2__2022-09-06_19-31-14", "handle-pull-v2__2022-09-06_19-31-23",
	# 							   "handle-pull-v2__2022-09-06_19-31-28", "handle-pull-v2__2022-09-06_19-31-33", ]],
	# 	"Push-Wall-V2 Medium": ["push-wall-v2__2022-09-06_19-31-48", "push-wall-v2__2022-09-06_19-31-59",
	# 							"push-wall-v2__2022-09-06_19-32-09", "push-wall-v2__2022-09-06_19-32-20", ],
	# 	"Window-Open-V2 Medium": ["window-open-v2__2022-09-06_19-34-38", "window-open-v2__2022-09-06_19-34-47",
	# 							  "window-open-v2__2022-09-06_19-34-54", ]
	# }
	# i=0
	# for name in exp_dict.keys():
	# 	main(name)
	# 	i+=1
	# 	if i>=12:
	# 		break
