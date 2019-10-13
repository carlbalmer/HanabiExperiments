import gym
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.logging import logger


class CartPoleEnv(Env):

    def __init__(self):
        self._env = gym.make('CartPole-v0')
        self._action_space = IntBox(low=0, high=self._env.action_space.n, dtype="int16")

    def reset(self):
        return self._env.reset()

    def step(self, action):
        observation, reward, done, _ = self._env.step(action)
        return EnvStep(observation, reward, done, None)
        # env_info is only sometimes not None (in the case when the agent reaches >199 steps)
        # this causes problems so it is set to be always none

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def horizon(self):
        return self._env._max_episode_steps


class CartPoleMixin:

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(input_size=env_spaces.observation.shape[0],
                    output_size=env_spaces.action.n)