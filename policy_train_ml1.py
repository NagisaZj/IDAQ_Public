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
    ml1 = metaworld.ML1(variant['env_name'],seed=1337)  # Construct the benchmark, sampling tasks

    env = ml1.train_classes[variant['env_name']]()  # Create an environment with task
    # print(ml1.train_tasks)
    env.train_tasks = ml1.train_tasks
    task = random.choice(ml1.train_tasks)
    env.set_task(task)

    tasks = list(range(len(env.train_tasks)))
    env=gym.wrappers.TimeLimit(gym.wrappers.ClipAction(MetaWorldWrapper(env)),500)

    env.reset_task(goal_idx)



    if "cuda" in cfg.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(goal_idx % variant['util_params']['num_gpus'])
    os.environ["OMP_NUM_THREADS"] = str(16)
    os.environ["KMP_AFFINITY"] = "compact,granularity\=fine"
    # NOTE: for new environment variable to be effective, torch should be imported after assignment
    from rlkit.torch.sac.pytorch_sac.train import Workspace
    workspace = Workspace(cfg=cfg, env=env, env_name=variant['env_name'], goal_idx=goal_idx)
    if eval:
        print('evaluate:')
        workspace.run_evaluate()
    else:
        workspace.run()


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
    p = mp.Pool(min(mp.cpu_count(), num_gpus))
    if variant['env_params']['n_tasks'] > 1:
        p.starmap(experiment, product([variant], [cfg], random_task_id))
    else:
        experiment(variant=variant, cfg=cfg, goal_idx=goal_idx)


if __name__ == '__main__':
    #add a change 
    main()
