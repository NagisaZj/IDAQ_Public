import numpy as np
import matplotlib.pyplot as plt
import os

def plot(dataset_path,data_path,epoches,name):
    returns = []
    states = []
    actions = []
    rewards = []
    next_states = []
    path = dataset_path
    for i in range(1):
        for j in range(45):
            file_name = path+'/goal_idx%d'%i+'/trj_evalsample%d_step49500.npy'%j if 'point' not in path else path+'/goal_idx%d'%i+'/trj_evalsample%d_step4800.npy'%j
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



    # epoches = ["130","110","296","295","292","285","282","280","277","275","266","263","259","257","250","248","245"]
    dists = []
    returns = []

    epoches = []
    cnt = 0
    for i in range(300,0,-1):
        if os.path.exists(data_path+str(i)+'-run0.pkl'):
            epoches.append(str(i))
            cnt+=1
            if cnt==15:
                break

    for e in epoches:
        # print()
        file = data_path+e+'-run0.pkl'

        traj = np.load(file,allow_pickle=True)
        # print(traj)

        for t in range(5):
            current_traj = traj[t]
            returns.append(np.sum(current_traj['rewards']))
            print(t,np.sum(current_traj['rewards']))
            # dists.append(5)
            traj_sar = np.concatenate([current_traj['observations'].flatten(),current_traj['actions'].flatten(),current_traj['rewards'].flatten(),],-1)
            print(traj_sar.shape)
            traj_sas = np.concatenate([current_traj['observations'].flatten(),current_traj['actions'].flatten(),current_traj['next_observations'].flatten(),],-1)
            # print(traj_sar.shape,sar.shape)
            r_dist = find_min(traj_sar,sar)
            s_dist = find_min(traj_sas,sas)
            print(r_dist+s_dist,r_dist,s_dist)
            dists.append(r_dist+s_dist)



    # file = '/data2/zj/Offline-MetaRL/output/push-v2__2023-01-20_14-28-50/push-v2/debug/eval_trajectories/task0-epoch110-run0.pkl'
    #
    # traj = np.load(file,allow_pickle=True)
    # # print(traj)
    #
    # dists = []
    # returns = []
    # for t in range(len(traj)):
    #     current_traj = traj[t]
    #     returns.append(np.sum(current_traj['rewards']))
    #     print(t,np.sum(current_traj['rewards']))
    #     traj_sar = np.concatenate([current_traj['observations'],current_traj['actions'],current_traj['rewards'],],-1)
    #     print(traj_sar.shape)
    #     traj_sas = np.concatenate([current_traj['observations'],current_traj['actions'],current_traj['next_observations'],],-1)
    #     r_dist = find_min(traj_sar,sar)
    #     s_dist = find_min(traj_sas,sas)
    #     print(r_dist+s_dist,r_dist,s_dist)
    #     dists.append(r_dist+s_dist)
    #
    #
    # file2 = '/data2/zj/Offline-MetaRL/output/push-v2__2023-01-20_14-28-50/push-v2/debug/eval_trajectories/task0-epoch130-run0.pkl'
    #
    # traj = np.load(file2,allow_pickle=True)
    # # print(traj)
    #
    #
    # for t in range(len(traj)):
    #     current_traj = traj[t]
    #     returns.append(np.sum(current_traj['rewards']))
    #     print(t,np.sum(current_traj['rewards']))
    #     traj_sar = np.concatenate([current_traj['observations'],current_traj['actions'],current_traj['rewards'],],-1)
    #     print(traj_sar.shape)
    #     traj_sas = np.concatenate([current_traj['observations'],current_traj['actions'],current_traj['next_observations'],],-1)
    #     r_dist = find_min(traj_sar,sar)
    #     s_dist = find_min(traj_sas,sas)
    #     print(r_dist+s_dist,r_dist,s_dist)
    #     dists.append(r_dist+s_dist)



    dataset_returns = []
    path = dataset_path
    for i in range(50):
        for j in range(45):
            file_name = path+'/goal_idx%d'%i+'/trj_evalsample%d_step49500.npy'%j if 'point' not in path else path+'/goal_idx%d'%i+'/trj_evalsample%d_step4800.npy'%j
            traj = np.load(file_name,allow_pickle=1)
            dataset_returns.append(sum(s[2] for s in traj))

    if 'point' not in path:

        return_plots = np.zeros(500)
        for r in dataset_returns:
            idx = int(np.floor(r/10))
            return_plots[idx]+=1
        return_plots /=len(dataset_returns)

    else:
        return_plots = np.zeros(30)
        for r in dataset_returns:
            idx = int(np.floor(r / -1))
            return_plots[idx] += 1
        return_plots /= len(dataset_returns)

    new_returns = np.array(returns)
    position = int(new_returns.shape[0]*0.8)
    new_returns = np.sort(new_returns)

    plt.figure()
    plt.title('Return-based quantification, '+name,fontsize=15)
    plt.xlabel('Episode Return',fontsize=15)
    plt.ylabel('Min Distance',fontsize=15)
    # max_length = int(np.max(dists))+1
    max_length = 5
    if 'point' not in path:
        for i in range(500):
            print(return_plots[i],i)
            plt.plot(np.ones(max_length)*(10*i+5),np.arange(max_length)/4*4.5,alpha=np.clip(return_plots[i]*3,0,0.5),linewidth=5,color='r')
    else:
        for i in range(30):
            print(return_plots[i],i)
            plt.plot(np.ones(max_length)*(i*-1),np.arange(max_length)/4*4.5,alpha=np.clip(return_plots[i]*3,0,0.5),linewidth=5,color='r')
    # plt.plot(np.ones(max_length)*new_returns[position],np.arange(max_length)/4*4.5,'-',alpha=0.3,linewidth=5,color='g')
    plt.scatter(returns,dists,alpha=1)


if __name__=='__main__':
    dataset_paths = ['./data/push-v2','./data/pick-place-v2','./data/reach-v2','./data/soccer-v2','./data/drawer-close-v2','./data/sparse-point-robot',]
    data_paths = ['/data2/zj/Offline-MetaRL/output/push-v2__2023-01-20_14-28-50/push-v2/debug/eval_trajectories/task0-epoch',
         '/data2/zj/Offline-MetaRL/output/pick-place-v2__2023-01-22_16-56-53/pick-place-v2/debug/eval_trajectories/task0-epoch','/data2/zj/Offline-MetaRL/output/reach-v2__2023-01-22_21-21-22/reach-v2/debug/eval_trajectories/task0-epoch','/data2/zj/Offline-MetaRL/output/soccer-v2__2023-01-22_21-20-45/soccer-v2/debug/eval_trajectories/task0-epoch','/data2/zj/Offline-MetaRL/output/drawer-close-v2__2023-01-22_21-22-14/drawer-close-v2/debug/eval_trajectories/task0-epoch',
                  # '/data2/zj/Offline-MetaRL/output/sparse-point-robot__2023-01-22_21-22-30/sparse-point-robot/debug/eval_trajectories/task0-epoch'
                  ]
    epochess = [["130","110","296","295","292","285","282","280","277","275","266","263","259","257","250","248","245"],
                ["164","163","162","161","160","159","158","159","157","156","155","154","153","152","151","150","149"],
                ["125","120","119","103","99","97","94","92","91","85","81","74","69","50","45","38","32"],
                ["122","121","120","118","116","113","112","109","106","105","104","103","102","101","99","98","97"],
                ["108","100","93","91","86","80","79","74","72","69","66","65","49","45","44","43","42"],
                ["48","47","38","34","32","28","25","21","19","10"],]
    names = ['Push-V2','Pick-Place-V2','Reach-V2','Soccer-V2','Drawer-Close-V2','Point-Robot']

    dataset_paths = [
        './data/push-v2', './data/pick-place-v2', './data/reach-v2', './data/soccer-v2',
                     './data/drawer-close-v2', './data/sweep-v2med', './data/peg-insert-side-v2med',
        './data/sparse-point-robot',  ]
    data_paths = [
        '/data2/zj/Offline-MetaRL/output/push-v2__2023-01-20_14-28-50/push-v2/debug/eval_trajectories/task0-epoch',
        '/data2/zj/Offline-MetaRL/output/pick-place-v2__2023-01-22_16-56-53/pick-place-v2/debug/eval_trajectories/task0-epoch',
        '/data2/zj/Offline-MetaRL/output/reach-v2__2023-01-22_21-21-22/reach-v2/debug/eval_trajectories/task0-epoch',
        '/data2/zj/Offline-MetaRL/output/soccer-v2__2023-01-22_21-20-45/soccer-v2/debug/eval_trajectories/task0-epoch',
        '/data2/zj/Offline-MetaRL/output/drawer-close-v2__2023-01-22_21-22-14/drawer-close-v2/debug/eval_trajectories/task0-epoch',
        '/data2/zj/Offline-MetaRL/output/sweep-v2__2023-01-25_10-26-17/sweep-v2/debug/eval_trajectories/task0-epoch',
        '/data2/zj/Offline-MetaRL/output/peg-insert-side-v2__2023-01-25_10-35-30/peg-insert-side-v2/debug/eval_trajectories/task0-epoch',
        '/data2/zj/Offline-MetaRL/output/sparse-point-robot__2023-01-22_21-22-30/sparse-point-robot/debug/eval_trajectories/task0-epoch',
        ]
    epochess = [
        ["130", "110", "296", "295", "292", "285", "282", "280", "277", "275", "266", "263", "259", "257", "250", "248",
         "245"],
        # ["299", "298", "297", "296", "295", "294", "293", "292", "291", "290", "289", "288", "287", "286", "285", "284",
        #  "283"],
        ["295", "293", "286", "279", "273", "272", "268", "255", "248", "241", "240", "239", "238", "217", "216", "215",
         "213"],
        ["291", "282", "277", "267", "266", "263", "255", "254", "242", "241", "223", "221", "213", "212", "199", "193", "186"],
        ["298", "292", "290", "289", "288", "285", "281", "278", "276", "271", "270", "269", "261", "260", "258", "257",
         "256"],
        ["108", "100", "93", "91", "86", "80", "79", "74", "72", "69", "66", "65", "49", "45", "44", "43", "42"],
        ["108", "100", "93", "91", "86", "80", "79", "74", "72", "69", "66", "65", "49", "45", "44", "43", "42"],
        ["108", "100", "93", "91", "86", "80", "79", "74", "72", "69", "66", "65", "49", "45", "44", "43", "42"],
    ["48","47","38","34","32","28","25","21","19","10"],]
    names = [
        'Push-V2', 'Pick-Place-V2', 'Reach-V2', 'Soccer-V2', 'Drawer-Close-V2', 'Sweep-V2 Medium', 'Peg-Insert-Side-V2 Medium',
        "Point-Robot"]
    for i in range(len(data_paths)):
        plot(dataset_paths[i],data_paths[i],epochess[i],names[i])


    plt.show()