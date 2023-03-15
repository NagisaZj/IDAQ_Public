import numpy as np
import matplotlib.pyplot as plt
import os,pickle,matplotlib





show_tasks_ids=[51,24,49,74,99]
show_tasks_ids=[2]

colors = ['r','b','g','purple','y']

figure = plt.figure(figsize=(20, 6))

name='push-v2__2023-01-19_12-02-15'
epoch = 29
# name='sparse-point-robot__2022-11-12_09-31-49'
# epoch = 28
gr = 0.2  # goal radius, for visualization purposes
run_num = 0
cmap = matplotlib.cm.get_cmap('plasma')
sample_locs = np.linspace(0, 0.9, 20)
colors = [cmap(s) for s in sample_locs]

expdir = 'output/' + name + '/push-v2/debug/eval_trajectories/'
test_task_list = [show_tasks_ids[0]]


def load_pkl(task):
    with open(os.path.join(expdir, 'task{}-epoch{}-run{}.pkl'.format(task, epoch, run_num)), 'rb') as f:
        data = pickle.load(f)
    return data

def load_pkl_std(task):
    with open(os.path.join(expdir, 'std-task{}-epoch{}-run{}.pkl'.format(task, epoch, run_num)), 'rb') as f:
        data = pickle.load(f)
    return data


def load_pkl_prior():
    with open(os.path.join(expdir, 'prior-epoch{}.pkl'.format(epoch)), 'rb') as f:
        data = pickle.load(f)
    return data




all_paths = []
for task in test_task_list:
    paths = [t['observations'] for t in load_pkl(task)]
    all_paths.append(paths)

all_paths_returns = []
for task in test_task_list:
    paths = [np.sum(t['rewards']) for t in load_pkl(task)]
    all_paths_returns.append(paths)

all_paths_pes = []
for task in test_task_list:
    paths = [np.mean(t['prediction_errors']) for t in load_pkl(task)]
    print(len(paths),paths)
    all_paths_pes.append(paths)

all_paths_pe_stds = []
for task in test_task_list:
    paths = [np.std(t['prediction_errors']) for t in load_pkl(task)]
    print(len(paths),paths)
    all_paths_pe_stds.append(paths)

all_paths_uncertainties = []
for task in test_task_list:
    paths = [np.sum(t['uncertanties']) for t in load_pkl(task)]
    all_paths_uncertainties.append(paths)
all_paths_uncertainties[0]=np.array(all_paths_uncertainties[0])*-1
min1,max1 = np.min(all_paths_uncertainties[0]),np.max(all_paths_uncertainties[0])
# all_paths_uncertainties = (all_paths_uncertainties-min1)/(max1-min1)
argmin = np.argmin(all_paths_uncertainties[0][:10])
from matplotlib import cm
map_vir = cm.get_cmap(name='viridis')

plt.figure()
print((np.array(all_paths_returns).shape))
print(all_paths_returns,all_paths_uncertainties)
plt.scatter(np.array(all_paths_returns),np.array(all_paths_uncertainties))
plt.show()
#
# plt.figure()
# plt.scatter(np.random.rand(10),np.random.rand(10),all_paths_uncertainties[0][:10],cmap='viridis')
# plt.colorbar()
# plt.show()

m=0
# plt.title('Episode %d, Return %.2f'%(m+11,all_paths_returns[0][m+10]))
# for g in goals:
#     # circle = plt.Circle((g[0], g[1]), radius=0.1, alpha=0.4)
#     # axes.add_artist(circle)
#     plt.plot(g[0], g[1],'*',markersize=20,color=np.array([1.0, 0.49, 0.0]),alpha=0.8)
#
# # with open('./data/pkl', 'rb') as f:
# #     data = pickle.load(f)
# for i in range(0,20):#np.array([0.9, 0.17, 0.31])
#     if i < 10 :
#         states = all_paths[0][i]
#         plt.plot(states[:-1, 0], states[:-1, 1],':' if i!=argmin else '', color= map_vir(all_paths_uncertainties[0][i]) if i<10 else np.array([0.9, 0.17, 0.31]),linewidth=5 if i<10 else 2)
#         #
#         # plt.plot(states[:-1, 0], states[:-1, 1],':' if i!=7 else '',color='g' if i!=7 else np.array([0.9, 0.17, 0.31]),linewidth=5 if i<10 else 2)
#
# # plt.plot(states[-1, 0], states[-1, 1], '-x', markersize=10, color=colors[0])
# # plt.set(aspect='equal')
# #
# # for i in range(10,20):
# #     states = all_paths[0][i]
# #     plt.plot(states[:-1, 0], states[:-1, 1], color='b',linewidth=3)
# for i in range(1):
#     id = show_tasks_ids[i]
#     color = colors[i]
#     positions = np.zeros((45*20,2))
#     for j in range(45):
#         traj = np.load('./data/sparse-point-robot/goal_idx%d/trj_evalsample%d_step4800.npy'%(id,j),allow_pickle=1)
#         print(i,j)
#         for t in range(20):
#             positions[t+j*20] = traj[t][0]
#             # if traj[t][0][0] !=0:
#             #     positions[t+j*20]+=np.random.rand(2) * 0.05
#     plt.scatter(positions[:,0],positions[:,1],s=20,color=np.array([0.19, 0.55, 0.91]),alpha=0.05)
# plt.xlim(-1.5,1.5)
# plt.ylim(-0.5,1.5)
# # plt.colorbar('viridis')
# plt.xticks([])
# plt.yticks([])
# plt.show()



# fig=plt.figure()

for m in range(20):
    plt.subplot(4,5,m+1)
    # plt.title('Episode %d,\n Pridiction Error Mean %.2e,\n Prediciton Error Std %.2e'%(m+1,all_paths_pes[0][m+0],all_paths_pe_stds[0][m+0]))

    plt.title('Episode %d,\n Uncertainty %.2e,\n Prediciton Error Std %.2e'%(m+1,all_paths_uncertainties[0][m+0],all_paths_pe_stds[0][m+0]))
    #
    # plt.title('Episode %d, Uncertainty %.3f'%(m+1,all_paths_uncertainties[0][m+0]))
    # plt.title('Episode %d, Return %.2f' % (m + 1, all_paths_returns[0][m + 0]))
    for g in goals:
        # circle = plt.Circle((g[0], g[1]), radius=0.1, alpha=0.4)
        # axes.add_artist(circle)
        plt.plot(g[0], g[1],'*',markersize=10,color=np.array([1.0, 0.49, 0.0]),alpha=0.8)

# with open('./data/pkl', 'rb') as f:
#     data = pickle.load(f)
#     for i in range(0, 20):
#         if i < 10 or i == (m + 10):
#             states = all_paths[0][i]
#             plt.plot(states[:-1, 0], states[:-1, 1], ':' if i < 10 else '',
#                      color='g' if i < 10 else np.array([0.9, 0.17, 0.31]), linewidth=3 if i < 10 else 2)
    for i in range(0,20):#np.array([0.9, 0.17, 0.31])
        if i ==m+0:
            print(argmin)
            states = all_paths[0][i]
            plt.plot(states[:-1, 0], states[:-1, 1],':' if i!=0 else '', color= 'g' if i !=0 else np.array([0.9, 0.17, 0.31]),linewidth=3if i<10 else 2)
    # plt.plot(states[-1, 0], states[-1, 1], '-x', markersize=10, color=colors[0])
    # plt.set(aspect='equal')
#
# for i in range(10,20):
#     states = all_paths[0][i]
#     plt.plot(states[:-1, 0], states[:-1, 1], color='b',linewidth=3)
    for i in range(1):
        id = show_tasks_ids[i]
        color = colors[i]
        positions = np.zeros((45*20,2))
        for j in range(45):
            traj = np.load('./data/sparse-point-robot/goal_idx%d/trj_evalsample%d_step4800.npy'%(id,j),allow_pickle=1)
            print(i,j)
            for t in range(20):
                positions[t+j*20] = traj[t][0]
                if traj[t][0][0] !=0:
                    positions[t+j*20]+=np.random.rand(2) * 0.05
        plt.scatter(positions[:,0],positions[:,1],s=1,color=np.array([0.19, 0.55, 0.91]),alpha=0.05)
    plt.xlim(-1.5,1.5)
    plt.ylim(-0.5,1.5)

# plt.legend(prop={'size': 15.0},)
# plt.tick_params('x', labelsize=15.0)
# plt.tick_params('y', labelsize=15.0)
plt.show()


