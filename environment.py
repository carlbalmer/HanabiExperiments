import gym
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox


class CartPoleEnv(Env):
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self._action_space = IntBox(low=0, high=self.env.action_space.n - 1, dtype="int16")
        self._observation_space = IntBox(shape=self.env.observation_space.shape, dtype="float32")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return EnvStep(*self.env.step(action))

    @property
    def horizon(self):
        return self.env._max_episode_steps