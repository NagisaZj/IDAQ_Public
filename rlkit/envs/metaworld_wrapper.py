import numpy as np
from gym import Env
from gym.spaces import Box
# import mujoco_py
import gym
from rlkit.core.serializable import Serializable



class MetaWorldWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._goal_idx = 0
        self.train_tasks = env.train_tasks
        env.set_task(self.train_tasks[self._goal_idx])
        self._goal = self.train_tasks[self._goal_idx]


    def reset_task(self, idx):
        self._goal_idx = idx
        self.env.set_task(self.train_tasks[self._goal_idx])
        self._goal = self.train_tasks[self._goal_idx]
        return self.env.reset()

    def get_all_task_idx(self):
        return range(len(self.train_tasks))




