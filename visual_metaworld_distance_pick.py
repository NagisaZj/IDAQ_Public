import numpy as np
import matplotlib.pyplot as plt



returns = []
states = []
actions = []
rewards = []
next_states = []
path = './data/pick-place-v2'
for i in range(5,6):
    for j in range(45):
        file_name = path+'/goal_idx%d'%i+'/trj_evalsample%d_step49500.npy'%j
        traj = np.load(file_name,allow_pickle=1)
        # for t in traj:
        #  states.append(t[0])
        #  actions.append(t[1])
        #  rewards.append(t[2])
        #  next_states.append(t[3])
        # print(np.array([s[0] for s in traj]).shape)
        states.append(np.array([s[0] for s in traj]).flatten())
        actions.append(np.array([s[1] for s in traj]).flatten())
        rewards.append(np.array([s[2] for s in traj]).flatten())
        next_states.append(np.array([s[3] for s in traj]).flatten())
states=np.array(states)
actions=np.array(actions)
rewards=np.array(rewards)
next_states=np.array(next_states)
# print(states.shape,rewards.shape)
sar = np.concatenate([states,actions,rewards],1)
sas = np.concatenate([states,actions,next_states],1)


# def find_min(mine,dataset):
#     total_distance = 0
#     for t in range(mine.shape[0]):
#         data = mine[0,:]
#         distance = np.linalg.norm(dataset-data,ord=2,axis=-1,keepdims=False)
#         total_distance += np.min(distance) / np.linalg.norm(dataset[np.argmin(distance),:],ord=2)
#     return total_distance / mine.shape[0]


def find_min(mine,dataset):
    total_distance = 0

    distance = np.linalg.norm(dataset-mine,ord=2,axis=-1,keepdims=False)
    total_distance += np.min(distance) / np.linalg.norm(dataset[np.argmin(distance),:],ord=2)
    return total_distance


print(sar.shape)

epoches = ["135","132","124","123","117","113","153","155","156","160","168","173","180","138","137"]
dists = []
returns = []
for e in epoches:

    file = '/data2/zj/Offline-MetaRL/output/pick-place-v2__2023-01-21_14-38-13/pick-place-v2/debug/eval_trajectories/task5-epoch'+e+'-run0.pkl'

    traj = np.load(file,allow_pickle=True)
    # print(traj)

    for t in range(10):
        current_traj = traj[t]
        returns.append(np.sum(current_traj['rewards']))
        print(t,np.sum(current_traj['rewards']))
        # dists.append(5)
        traj_sar = np.concatenate([current_traj['observations'].flatten(),current_traj['actions'].flatten(),current_traj['rewards'].flatten(),],-1)
        print(traj_sar.shape)
        traj_sas = np.concatenate([current_traj['observations'].flatten(),current_traj['actions'].flatten(),current_traj['next_observations'].flatten(),],-1)
        r_dist = find_min(traj_sar,sar)
        s_dist = find_min(traj_sas,sas)
        print(r_dist+s_dist,r_dist,s_dist)
        dists.append(r_dist+s_dist)


# for e in epoches:
#
#     file = '/data2/zj/Offline-MetaRL/output/pick-place-v2__2023-01-21_14-38-13/pick-place-v2/debug/eval_trajectories/task5-epoch'+e+'-run0.pkl'
#
#     traj = np.load(file,allow_pickle=True)
#     # print(traj)
#
#     for t in range(5):
#         current_traj = traj[t]
#         returns.append(np.sum(current_traj['rewards']))
#         print(t,np.sum(current_traj['rewards']))
#         # dists.append(5)
#         traj_sar = np.concatenate([current_traj['observations'],current_traj['actions'],current_traj['rewards'],],-1)
#         print(traj_sar.shape)
#         traj_sas = np.concatenate([current_traj['observations'],current_traj['actions'],current_traj['next_observations'],],-1)
#         r_dist = find_min(traj_sar,sar)
#         s_dist = find_min(traj_sas,sas)
#         print(r_dist+s_dist,r_dist,s_dist)
#         dists.append(r_dist+s_dist)

dataset_returns = []
path = './data/pick-place-v2'
for i in range(50):
    for j in range(45):
        file_name = path+'/goal_idx%d'%i+'/trj_evalsample%d_step49500.npy'%j
        traj = np.load(file_name,allow_pickle=1)
        dataset_returns.append(sum(s[2] for s in traj))

return_plots = np.zeros(500)
for r in dataset_returns:
    idx = int(np.floor(r/10))
    return_plots[idx]+=1
return_plots /=len(dataset_returns)


plt.figure()
plt.scatter(returns,dists)
max_length = int(np.max(dists))+1
for i in range(500):
    print(return_plots[i],i)
    plt.plot(np.ones(max_length)*(10*i+5),np.arange(max_length)/4*4.5,alpha=return_plots[i]*10,linewidth=5,color='r')
plt.show()