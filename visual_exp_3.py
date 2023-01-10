import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import csv
import pickle
import os
import torch
import click
# config
def cal_rew(encoder,path):
    s,a,r = path['observations'],path['actions'],path['rewards']
    s,a,r=torch.FloatTensor(s),torch.FloatTensor(a),torch.FloatTensor(r)
    input = torch.cat([s,a,r],dim=1)
    print(input.shape)
    input = torch.unsqueeze(input,0)
    output = encoder.forward_seq(input)
    print(output.shape)
    var = torch.mean(torch.log(torch.nn.functional.softplus(output[:,:,10:])),dim=2)
    var = torch.mean(torch.nn.functional.softplus(output[:,:,10:]),dim=2)
    #var = torch.log(var)
    print(var.shape)
    #print(r,var)
    return var.view(r.shape[0],r.shape[1])

@click.command()
@click.option('--name', default='sparse-point-robot__2022-09-09_08-47-58')

def main(name):

    #tlow, thigh = 80, 100  # task ID range

    test_task_list = [0]
    run_num = 0
    task_num = len(test_task_list)

    # see `n_tasks` and `n_eval_tasks` args in the training config json
    # by convention, the test tasks are always the last `n_eval_tasks` IDs
    # so if there are 100 tasks total, and 20 test tasks, the test tasks will be IDs 81-100
    epoch = 49
    gr = 0.2  # goal radius, for visualization purposes

    expdir = 'output/' + name + '/sparse-point-robot/debug/eval_trajectories/'

    # expdir = './outputfin2/sparse-point-robot-noise/{}/eval_trajectories/'.format(exp_id)
    # dir = './outputfin2/sparse-point-robot-noise/{}/'.format(exp_id)
    # helpers
    def load_pkl(task):
        with open(os.path.join(expdir, 'task{}-epoch{}-run{}.pkl'.format(task, epoch, run_num)), 'rb') as f:
            data = pickle.load(f)
        return data

    def load_pkl_prior():
        with open(os.path.join(expdir, 'prior-epoch{}.pkl'.format(epoch)), 'rb') as f:
            data = pickle.load(f)
        return data

    goals = [load_pkl(task)[0]['goal'] for task in test_task_list]

    plt.figure(figsize=(60, 160))
    axes = plt.axes()
    axes.set(aspect='equal')
    plt.axis([-1.55, 1.55, -0.55, 1.55])
    for g in goals:
        circle = plt.Circle((g[0], g[1]), radius=gr)
        axes.add_artist(circle)
    rewards = 0
    final_rewards = 0
    '''for traj in paths:
        rewards += sum(traj['rewards'])
        final_rewards += traj['rewards'][-1]
        states = traj['observations']
        plt.plot(states[:-1, 0], states[:-1, 1], '-o')
        plt.plot(states[-1, 0], states[-1, 1], '-x', markersize=20)'''

    num_trajs = 20

    all_paths = []
    for task in test_task_list:
        paths = [t['observations'] for t in load_pkl(task)]
        all_paths.append(paths)

    # color trajectories in order they were collected
    cmap = matplotlib.cm.get_cmap('plasma')
    sample_locs = np.linspace(0, 0.9, num_trajs)
    colors = [cmap(s) for s in sample_locs]

    fig, axes = plt.subplots(4, num_trajs // 4, figsize=(12, 20))
    t = 0

    all_paths_rew = []
    for task in test_task_list:
        paths = [t['rewards'] for t in load_pkl(task)]
        all_paths_rew.append(paths)

    '''all_paths_z_means = []
    all_paths_z_vars = []
    for task in range(tlow, thigh):
        means = [t['z_means'] for t in load_pkl(task)]
        vars = [t['z_vars'] for t in load_pkl(task)]
        all_paths_z_means.append(means)
        all_paths_z_vars.append(means)'''

    reward = np.zeros((task_num, 1))
    final_rew = np.zeros((task_num, 1))
    for m in range(task_num):
        for n in range(len(all_paths_rew[m])):
            reward[m] = reward[m] + np.sum(all_paths_rew[m][n])
            # reward[m] = reward[m] + all_paths_rew[m][n][-1]
        reward[m] = reward[m] / len(all_paths_rew[m])
    reward = reward
    # print(reward)
    # print(np.mean(reward))

    count = 0
    for i in range(4):
        for j in range(num_trajs // 4):
            axes[i, j].set_xlim([-2.05, 2.05])
            axes[i, j].set_ylim([-1.05, 2.05])
            for k, g in enumerate(goals):
                alpha = 0.2
                circle = plt.Circle((g[0], g[1]), radius=gr, alpha=alpha)
                axes[i, j].add_artist(circle)

            states = all_paths[t][count]
            axes[i, j].plot(states[:-1, 0], states[:-1, 1], '-', color=colors[count])
            axes[i, j].plot(states[-1, 0], states[-1, 1], '-x', markersize=10, color=colors[count])
            axes[i, j].set(aspect='equal')

            axes[i, j].set_title("Return:%.1f" % np.sum(all_paths_rew[t][count]))

            count += 1

    fig.suptitle("iteration: %d, task %d, run: %d, average reward of all tasks:%f" % (epoch, test_task_list[0], run_num, np.mean(reward)))

    #task = 1
    #fig, axes = plt.subplots(2, 4)

    # print(ap[0]['z_vars'])
    '''
    for m in range(1):
        for n in range(4):
            id = m * 4 + n
            axes[m, n].set_xlim([-2.05, 2.05])
            axes[m, n].set_ylim([-1.05, 2.05])
            for k, g in enumerate(goals):
                alpha = 1 if k == task else 0.2
                circle = plt.Circle((g[0], g[1]), radius=gr, alpha=alpha)
                axes[m, n].add_artist(circle)
            states = all_paths[task][id]
            print(states.shape)
            # rew = cal_rew(encoder, ap[id])
            axes[m, n].plot(states[:-1, 0], states[:-1, 1], '-', color=colors[0])
            axes[m, n].plot(states[-1, 0], states[-1, 1], '-x', markersize=10, color=colors[0])
            # axes[m, n].text(states[0, 0], states[0, 1], '%.3f\n%.3f' % (np.mean(ap[id]['z_means'][0]),np.min(ap[id]['z_vars'][0]**0.5)),fontsize=12)
            # axes[m, n].text(states[3, 0], states[3, 1],
            #                '%.3f\n%.3f' % (np.mean(ap[id]['z_means'][3]), np.min(ap[id]['z_vars'][3] ** 0.5)))
            # axes[m, n].text(states[-1, 0], states[-1, 1], '%.3f\n%.3f' % (np.mean(ap[id]['z_means'][-1]),np.min(ap[id]['z_vars'][-1]**0.5)),fontsize=12)
    '''
    # for i in range(10):
    #    print(ap[i]['z_means'])
    plt.savefig("figures/heatmaps/" + name + "_task" + str(test_task_list[0]) + "_epoch" + str(epoch) + "_run" + str(run_num) + ".png")

if __name__ =="__main__":
    main()
