import gym
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox


class CartPoleEnv(Env):

    def __init__(self):
        self._env = gym.make('CartPole-v0')
        self._action_space = IntBox(low=0, high=self._env.action_space.n, dtype="int16")
        #self._observation_space = IntBox(shape=self._env.observation_space.shape, dtype="float32")

    def reset(self):
        return self._env.reset()

    def step(self, action):
        observation, reward, done, env_info = self._env.step(action)
        if not env_info:
            env_info = None
        return EnvStep(observation, reward, done, env_info)

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