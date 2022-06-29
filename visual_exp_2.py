import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import csv
import pickle
import os
import colour
import torch
import click
from rlkit.torch.networks import SnailEncoder,MlpEncoder
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
@click.option('--name', default='sparse-point-robot__2022-06-28_23-15-44')

def main(name):

    tlow, thigh = 80, 100  # task ID range
    # see `n_tasks` and `n_eval_tasks` args in the training config json
    # by convention, the test tasks are always the last `n_eval_tasks` IDs
    # so if there are 100 tasks total, and 20 test tasks, the test tasks will be IDs 81-100
    epoch = 90
    gr = 0.3  # goal radius, for visualization purposes

    expdir = 'output/' + name + '/sparse-point-robot/debug/'

    # expdir = './outputfin2/sparse-point-robot-noise/{}/eval_trajectories/'.format(exp_id)
    # dir = './outputfin2/sparse-point-robot-noise/{}/'.format(exp_id)
    # helpers
    def load_pkl(task):
        with open(os.path.join(expdir, 'task{}-epoch{}-run0.pkl'.format(task, epoch)), 'rb') as f:
            data = pickle.load(f)
        return data

    def load_pkl_prior():
        with open(os.path.join(expdir, 'prior-epoch{}.pkl'.format(epoch)), 'rb') as f:
            data = pickle.load(f)
        return data

    goals = [load_pkl(task)[0]['goal'] for task in range(tlow, thigh)]

    plt.figure(figsize=(8, 8))
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

    mpl = 20
    num_trajs = 60

    all_paths = []
    for task in range(tlow, thigh):
        paths = [t['observations'] for t in load_pkl(task)]
        all_paths.append(paths)

    # color trajectories in order they were collected
    cmap = matplotlib.cm.get_cmap('plasma')
    sample_locs = np.linspace(0, 0.9, num_trajs)
    colors = [cmap(s) for s in sample_locs]

    fig, axes = plt.subplots(3, 3, figsize=(12, 20))
    t = 10

    all_paths_rew = []
    for task in range(tlow, thigh):
        paths = [t['rewards'] for t in load_pkl(task)]
        all_paths_rew.append(paths)

    '''all_paths_z_means = []
    all_paths_z_vars = []
    for task in range(tlow, thigh):
        means = [t['z_means'] for t in load_pkl(task)]
        vars = [t['z_vars'] for t in load_pkl(task)]
        all_paths_z_means.append(means)
        all_paths_z_vars.append(means)'''

    reward = np.zeros((20, 1))
    final_rew = np.zeros((20, 1))
    for m in range(20):
        for n in range(len(all_paths_rew[m])):
            reward[m] = reward[m] + np.sum(all_paths_rew[m][n])
            # reward[m] = reward[m] + all_paths_rew[m][n][-1]
        reward[m] = reward[m] / len(all_paths_rew[m])
    reward = reward
    # print(reward)
    # print(np.mean(reward))

    for j in range(3):
        for i in range(3):
            axes[i, j].set_xlim([-2.05, 2.05])
            axes[i, j].set_ylim([-1.05, 2.05])
            for k, g in enumerate(goals):
                alpha = 1 if k == t else 0.2
                circle = plt.Circle((g[0], g[1]), radius=gr, alpha=alpha)
                axes[i, j].add_artist(circle)
            indices = list(np.linspace(0, 4, num_trajs, endpoint=False).astype(np.int))
            counter = 0
            for idx in indices:
                states = all_paths[t][idx]
                axes[i, j].plot(states[:-1, 0], states[:-1, 1], '-', color=colors[counter])
                axes[i, j].plot(states[-1, 0], states[-1, 1], '-x', markersize=10, color=colors[counter])
                axes[i, j].set(aspect='equal')
                counter += 1
            axes[i, j].set_title("average reward:%f" % reward[t])
            t += 1

    fig.suptitle("iteration:%d, average reward of all tasks:%f" % (epoch, np.mean(reward)))

    task = 1
    fig, axes = plt.subplots(2, 4)

    # encoder.load_state_dict(torch.load(os.path.join(dir, 'context_encoder.pth')))
    ap = [t for t in load_pkl(task + 80)]

    # print(ap[0]['z_vars'])
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
    # for i in range(10):
    #    print(ap[i]['z_means'])
    plt.savefig("figures/heatmaps/" + name + "_" + str(epoch) + ".png")

if __name__ =="__main__":
    main()
