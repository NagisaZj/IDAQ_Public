import numpy as np
import matplotlib.pyplot as plt
import os,pickle,matplotlib

show_tasks_ids=[50,24,49,74,99]
colors = ['r','b','g','purple','y']

figure = plt.figure(figsize=(16, 9))
name = 'sparse-point-robot__2022-09-09_08-48-18'
epoch = 49
gr = 0.2  # goal radius, for visualization purposes
run_num = 0
cmap = matplotlib.cm.get_cmap('plasma')
sample_locs = np.linspace(0, 0.9, 20)
colors = [cmap(s) for s in sample_locs]

expdir = 'output/' + name + '/sparse-point-robot/debug/eval_trajectories/'
test_task_list = [show_tasks_ids[0]]


def load_pkl(task):
    with open(os.path.join(expdir, 'task{}-epoch{}-run{}.pkl'.format(task, epoch, run_num)), 'rb') as f:
        data = pickle.load(f)
    return data


def load_pkl_prior():
    with open(os.path.join(expdir, 'prior-epoch{}.pkl'.format(epoch)), 'rb') as f:
        data = pickle.load(f)
    return data


goals = [load_pkl(task)[0]['goal'] for task in test_task_list]
axes = plt.axes()
axes.set(aspect='equal')
# plt.axis([-1.55, 1.55, -0.55, 1.55])
num_trajs = 20

all_paths = []
for task in test_task_list:
    paths = [t['observations'] for t in load_pkl(task)]
    all_paths.append(paths)

for g in goals:
    # circle = plt.Circle((g[0], g[1]), radius=0.1, alpha=0.4)
    # axes.add_artist(circle)
    plt.plot(g[0], g[1],'*',markersize=40,color=np.array([1.0, 0.49, 0.0]),alpha=0.8)

# with open('./data/pkl', 'rb') as f:
#     data = pickle.load(f)
for i in range(0,10):
    states = all_paths[0][i]
    plt.plot(states[:-1, 0], states[:-1, 1],':' if i!=8 else '', color='g' if i!=8 else np.array([0.9, 0.17, 0.31]),linewidth=3)
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
    plt.scatter(positions[:,0],positions[:,1],s=100,color=np.array([0.19, 0.55, 0.91]),alpha=0.05)
plt.xlim(-1.5,2.0)
plt.ylim(-0.5,2.0)

# plt.legend(prop={'size': 15.0},)
plt.tick_params('x', labelsize=15.0)
plt.tick_params('y', labelsize=15.0)
plt.show()


