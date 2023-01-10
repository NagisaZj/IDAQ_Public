import random
import matplotlib.colors as mcolors
import numpy as np
import torch
import matplotlib.pyplot as plt

from gym import Env
from gym import spaces
from . import register_env
# from utils import helpers as utl

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@register_env('treasure-huntori')
class TreasureHunt(Env):
    def __init__(self,
                 max_episode_steps=100,
                 mountain_height=1,
                 treasure_reward=1,
                 timestep_penalty=-1,
                 ):

        # environment layout
        self.mountain_top = np.array([0, 0])
        self.start_position = np.array([0, -1])
        self.mountain_height = mountain_height
        self.treasure_reward = treasure_reward
        self.timestep_penalty = timestep_penalty
        # NOTE: much of the code assumes these radii, if you change them you need to make a few other changes!
        self.goal_radius = 1.0
        self.mountain_radius = 0.5
        # You can get around the full circle in 62 steps
        self._max_episode_steps = max_episode_steps

        # observation/action/task space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.task_dim = 2

        # initialise variables
        self.step_count = None
        self.goal = None
        self.state = None
        self.reset_task()
        self.reset()

    def reset(self):
        self.step_count = 0
        self.state = self.start_position
        return self._get_obs()

    def sample_task(self):
        # sample a goal from a circle with radius 1 around the mountain top [0, 0]
        angle = random.uniform(0, 2 * np.pi)
        goal = self.goal_radius * np.array((np.cos(angle), np.sin(angle)))
        return goal

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), action

        # execute action - make sure the agent does not walk outside [-1.5, 1.5] in any direction
        self.state = np.clip(self.state + 0.1 * action, -1.5, 1.5)
        done = False

        # if the agent is on the goal, it gets a high reward
        mountain_top_distance = np.linalg.norm(self.mountain_top - self.state, 2)
        treasure_distance = np.linalg.norm(self.goal - self.state, 2)

        # CASE: agent is on mountain - penalise (the higher the more penalty)
        if mountain_top_distance <= self.mountain_radius:
            reward = self.mountain_height * (- self.mountain_radius + mountain_top_distance) + self.timestep_penalty
        # CASE: agent is near the goal - give it the treasure reward
        elif treasure_distance <= 0.1:
            reward = self.treasure_reward
        # CASE: agent is somewhere else - make sure it doesn't walk too far away
        else:
            # make agent not walk too far outside circle
            dist_to_center = max([1, np.sqrt(self.state[0]**2 + self.state[1]**2)])
            reward = self.timestep_penalty * dist_to_center

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True

        ob = self._get_obs()
        info = {'task': self.get_task()}
        return ob, reward, done, info

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        return task

    def set_task(self, task):
        self.goal = task

    def get_task(self):
        return self.goal

    def _get_obs(self):
        agent_is_on_mountain = np.linalg.norm(self.state, 2) < 0.1
        if agent_is_on_mountain:
            obs = np.concatenate((self.state, self.goal))
        else:
            obs = np.concatenate((self.state, np.zeros(2)))
        return obs

    def render(self, mode='human'):
        pass


@register_env('treasure-hunt')
class MyTreasureHunt(Env):
    def __init__(self,
                 max_episode_steps=100,
                 mountain_height=1,
                 treasure_reward=1,
                 timestep_penalty=-1,
                 n_tasks=None,
                 randomize_tasks=True
                 ):

        # environment layout
        self.mountain_top = np.array([0, 0])
        self.start_position = np.array([0, -1])
        self.mountain_height = mountain_height
        self.treasure_reward = treasure_reward
        self.timestep_penalty = timestep_penalty
        # NOTE: much of the code assumes these radii, if you change them you need to make a few other changes!
        self.goal_radius = 1.0
        self.mountain_radius = 0.5
        # You can get around the full circle in 62 steps
        self._max_episode_steps = max_episode_steps

        # observation/action/task space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.task_dim = 2
        self.num_tasks = 100

        np.random.seed(1337)
        radius = 1.0
        angles = np.linspace(0, 2*np.pi, num=self.num_tasks)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        goals = np.stack([xs, ys], axis=1)
        if 1:
            np.random.shuffle(goals)
        goals = goals.tolist()

        self.goals = goals

        # initialise variables
        self.step_count = None
        self.goal = None
        self.state = None
        self.reset_task(0)
        self.reset()

    def reset(self):
        self.step_count = 0
        self.state = self.start_position
        return self._get_obs()


    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), action

        # execute action - make sure the agent does not walk outside [-1.5, 1.5] in any direction
        self.state = np.clip(self.state + 0.1 * action, -1.5, 1.5)
        done = False

        # if the agent is on the goal, it gets a high reward
        mountain_top_distance = np.linalg.norm(self.mountain_top - self.state, 2)
        treasure_distance = np.linalg.norm(self.goal - self.state, 2)

        # CASE: agent is on mountain - penalise (the higher the more penalty)
        if mountain_top_distance <= self.mountain_radius:
            reward = self.mountain_height * (- self.mountain_radius + mountain_top_distance) + self.timestep_penalty
        # CASE: agent is near the goal - give it the treasure reward
        elif treasure_distance <= 0.1:
            reward = self.treasure_reward
        # CASE: agent is somewhere else - make sure it doesn't walk too far away
        else:
            # make agent not walk too far outside circle
            dist_to_center = max([1, np.sqrt(self.state[0]**2 + self.state[1]**2)])
            reward = self.timestep_penalty * dist_to_center

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True

        ob = self._get_obs()
        info = {'task': self.get_task()}
        return ob, reward, done, info

    def reset_task(self, task_id=None):
        # if task is None:
        #     task = self.sample_task()
        self.set_task(task_id)
        return

    def set_task(self, task_id):
        self._goal_idx = task_id
        self.goal = self.goals[task_id]

    def get_task(self):
        return self.goal

    def _get_obs(self):
        agent_is_on_mountain = np.linalg.norm(self.state, 2) < 0.1
        if agent_is_on_mountain:
            obs = np.concatenate((self.state, self.goal))
        else:
            obs = np.concatenate((self.state, np.zeros(2)))
        return obs

    def render(self, mode='human'):
        pass