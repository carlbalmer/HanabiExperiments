import numpy
from gym.spaces import Box, Dict
from hanabi_learning_environment.rl_env import HanabiEnv
from ray.rllib import MultiAgentEnv

from rayExperiments.spaces import LegalActionDiscrete


class MultiAgentHanabiEnv(MultiAgentEnv):

    def __init__(self, env_config):
        self.env = HanabiEnv(env_config)
        self.state = self.env.reset()
        self.cum_reward = numpy.zeros((env_config["players"]), dtype=numpy.float)

        n_actions = 2 * env_config["hand_size"] + (env_config["colors"] + env_config["ranks"]) * (
                env_config["players"] - 1)
        self.action_space = LegalActionDiscrete(n_actions, self)

        sample_obs = self.reset()[0]
        self.observation_space = Dict({
            "board": Box(low=sample_obs["board"].min(),
                         high=sample_obs["board"].max(),
                         shape=sample_obs["board"].shape,
                         dtype=sample_obs["board"].dtype),
            "legal_actions": Box(
                low=sample_obs["legal_actions"].min(),
                high=sample_obs["legal_actions"].max(),
                shape=sample_obs["legal_actions"].shape,
                dtype=sample_obs["legal_actions"].dtype)
        })

    def reset(self):
        self.state = self.env.reset()
        self.cum_reward[:] = 0
        current_player, current_player_obs = self.extract_current_player_obs(self.state)
        return {current_player: current_player_obs}

    def step(self, action_dict):
        current_player, _ = self.extract_current_player_obs(self.state)
        current_player_action = action_dict[current_player]
        self.cum_reward[current_player] = 0
        assert self.action_space.contains(current_player_action)

        self.state, reward, done, _ = self.env.step(current_player_action.item())
        reward = reward / len(self.cum_reward) # scale the reward for each player - rllib sums up all player rewards
        self.cum_reward += reward
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
        current_player_obs = numpy.array(all_obs["player_observations"][current_player]["vectorized"], dtype=numpy.int)
        legal_actions = self.legal_actions_as_int_to_bool(
            all_obs["player_observations"][current_player]["legal_moves_as_int"])
        return current_player, {"board": current_player_obs, "legal_actions": legal_actions}

    def legal_actions_as_int_to_bool(self, legal_moves_as_int):
        return (numpy.in1d(numpy.arange(self.action_space.n), numpy.array(legal_moves_as_int))).astype(numpy.int)


HANABI_CONF_FULL_4p = {
    "colors": 5,
    "ranks": 5,
    "players": 4,
    "hand_size": 4,
    "max_information_tokens": 8,
    "max_life_tokens": 3,
    "observation_type": 1
}
