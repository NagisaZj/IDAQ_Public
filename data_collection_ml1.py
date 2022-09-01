"""
Training behavior policies for FOCAL

"""

import click
import json
import os

import gym, gym.wrappers
from hydra.experimental import compose, initialize

import argparse
import multiprocessing as mp
from multiprocessing import Pool
from itertools import product

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs.metaworld_wrapper import MetaWorldWrapper
from rlkit.envs import ENVS
from configs.default import default_config
import metaworld,random
import numpy as np
import metaworld.policies as p
import copy


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

initialize(config_dir="rlkit/torch/sac/pytorch_sac/config/")
cfg = compose("train.yaml")
def experiment(variant, cfg=cfg, goal_idx=0, seed=0,  eval=False):
    os.makedirs('./data/'+variant['env_name']+'/goal_idx%d'%goal_idx,exist_ok=True)
    ml1 = metaworld.MT1(variant['env_name'],seed=1337)  # Construct the benchmark, sampling tasks

    env = ml1.train_classes[variant['env_name']]()  # Create an environment with task
    # print(ml1.train_tasks)
    env.train_tasks = ml1.train_tasks
    task = random.choice(ml1.train_tasks)
    task = ml1.train_tasks[goal_idx]
    env.set_task(task)

    # tasks = list(range(len(env.train_tasks)))
    # env=gym.wrappers.TimeLimit(gym.wrappers.ClipAction(MetaWorldWrapper(env)),500)
    #
    # env.reset_task(goal_idx)


    if variant['env_name']=='push-v2':
        policy = p.SawyerPushV2Policy
    elif variant['env_name']=='reach-v2':
        policy = p.SawyerReachV2Policy
    elif variant['env_name']=='pick-place-v2':
        policy = p.SawyerPickPlaceV2Policy
    elif variant['env_name']=='push-wall-v2':
        policy = p.SawyerPushWallV2Policy
    elif variant['env_name']=='pick-place-wall-v2':
        policy = p.SawyerPickPlaceV2Policy
    elif variant['env_name']=='window-open-v2':
        policy = p.SawyerWindowOpenV2Policy
    elif variant['env_name']=='drawer-close-v2':
        policy = p.SawyerDrawerCloseV2Policy
    else:
        NotImplementedError


    success_cnt = 0
    while success_cnt <45:
        obs = env.reset()
        done = False
        episode_reward = 0
        trj = []
        step = 0
        success = 0
        while not done:
            # tmp_obs = copy.deepcopy(obs)
            action = policy.get_action(policy,obs)
            noise = np.random.randn(action.shape[0])  *0.1
            action = (action+noise).clip(-1,1)
            new_obs, reward, done, info = env.step(action)
            # env.render()
            done = float(1) if step+1==500 else done
            step +=1
            store_obs = copy.deepcopy(obs)
            store_new_obs = copy.deepcopy(new_obs)
            store_obs[-3:] = 0
            store_new_obs[-3:] = 0
            trj.append([store_obs, action, reward, store_new_obs])
            obs = new_obs
            episode_reward += reward
            success +=info['success']
        if 1:
            print(episode_reward,success,success_cnt)
            np.save(os.path.join('./data/'+variant['env_name']+'/goal_idx%d'%goal_idx, f'trj_evalsample{success_cnt}_step{49500}.npy'), np.array(trj))
            success_cnt+=1
        else:
            if np.random.rand()>0.9:
                print(episode_reward, success, success_cnt)
                np.save(os.path.join('./data/' + variant['env_name'] + '3/goal_idx%d' % goal_idx,
                                     f'trj_evalsample{success_cnt}_step{49500}.npy'), np.array(trj))
                success_cnt += 1

    return


@click.command()
@click.argument("config", default="./configs/sparse-point-robot.json")
@click.option("--num_gpus", default=8)
@click.option("--docker", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
@click.option("--eval", is_flag=True, default=False)
@click.option("--is_uniform", is_flag=True, default=False)
def main(config, num_gpus, docker, debug, eval, is_uniform, goal_idx=0, seed=0):
    variant = default_config
    cwd = os.getcwd()
    files = os.listdir(cwd)
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['num_gpus'] = num_gpus

    random_task_id = np.ndarray.tolist(np.random.permutation(variant['env_params']['n_tasks']))

    cfg.is_uniform = is_uniform
    print('cfg.is_uniform', cfg.is_uniform)

    #cfg.gpu_id = gpu
    #print('cfg.agent', cfg.agent)
    print(list(range(variant['env_params']['n_tasks'])))
    # multi-processing
    p = mp.Pool(min(mp.cpu_count(), 50))

    os.makedirs('./data/'+variant['env_name'],exist_ok=True)
    if variant['env_params']['n_tasks'] > 1:
        p.starmap(experiment, product([variant], [cfg], random_task_id))
    else:
        experiment(variant=variant, cfg=cfg, goal_idx=goal_idx)


if __name__ == '__main__':
    #add a change 
    main()
