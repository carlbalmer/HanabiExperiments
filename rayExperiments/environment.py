import numpy
from gym.spaces import Discrete, Box
from hanabi_learning_environment.rl_env import HanabiEnv
from ray.rllib import MultiAgentEnv


class MultiAgentHanabiEnv(MultiAgentEnv):

    def __init__(self, env_config):
        self.env = HanabiEnv(env_config)
        self.state = self.env.reset()
        self.cum_reward = numpy.zeros((env_config["players"]), dtype="int")

        n_actions = 2 * env_config["hand_size"] + (env_config["colors"] + env_config["ranks"]) * (
                env_config["players"] - 1)
        self.action_space = LegalActionDiscrete(n_actions, self)

        sample_obs = numpy.array(self.state["player_observations"][0]["vectorized"])
        self.observation_space = Box(low=sample_obs.min(), high=sample_obs.max(), shape=sample_obs.shape,
                                     dtype=sample_obs.dtype)

    def reset(self):
        self.state = self.env.reset()  #
        self.cum_reward[:] = 0
        current_player, current_player_obs = self.extract_current_player_obs(self.state)
        return {current_player: current_player_obs}

    def step(self, action_dict):
        current_player, _ = self.extract_current_player_obs(self.state)
        current_player_action = action_dict[current_player]
        self.cum_reward[current_player] = 0
        # assert self.action_space.contains(current_player_action)

        self.state, reward, done, _ = self.env.step(self.action_space.sample())  # self.env.step(current_player_action)
        self.cum_reward + reward
        next_player, next_player_obs = self.extract_current_player_obs(self.state)

        obs_dict = {next_player: next_player_obs}
        reward_dict = {next_player: self.cum_reward[next_player]}
        done_dict = {"__all__": done}
        if done:
            obs_dict = {player: next_player_obs for player in range(len(self.cum_reward))}
            reward_dict = {player: reward for player, reward in enumerate(self.cum_reward)}
        return obs_dict, reward_dict, done_dict, {}

    def extract_current_player_obs(self, all_obs):
        current_player = all_obs["current_player"]
        current_player_obs = numpy.array(all_obs["player_observations"][current_player]["vectorized"])
        return current_player, current_player_obs


class LegalActionDiscrete(Discrete):

    def __init__(self, n, env):
        super(LegalActionDiscrete, self).__init__(n)
        self.env = env

    def sample(self):
        legal_actions = self.env.state["player_observations"][self.env.state["current_player"]]["legal_moves_as_int"]
        return legal_actions[self.np_random.randint(len(legal_actions))]

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (numpy.generic, numpy.ndarray)) and (
                x.dtype.kind in numpy.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int in self.env.state["player_observations"][self.env.state["current_player"]]["legal_moves_as_int"]


HANABI_CONF_FULL_4p = {
    "colors": 5,
    "ranks": 5,
    "players": 4,
    "hand_size": 4,
    "max_information_tokens": 8,
    "max_life_tokens": 3,
    "observation_type": 1,
    #  "random_start_player": None,
    # "seed": None
}
