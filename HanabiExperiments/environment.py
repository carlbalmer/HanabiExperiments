import numpy
from gym.spaces import Box, Dict
from hanabi_learning_environment.rl_env import HanabiEnv
from ray.rllib import MultiAgentEnv

from HanabiExperiments.spaces import LegalActionDiscrete


class MultiAgentHanabiEnv(MultiAgentEnv):

    def __init__(self, env_config):
        self.env = HanabiEnv(env_config)
        self.state = self.env.reset()
        self.env_config = env_config
        self.cum_reward = numpy.zeros((env_config["players"]), dtype=numpy.float)
        self.last_action = numpy.zeros((env_config["players"]), dtype=numpy.int)
        self.last_action[:] = -1
        self.n_players = env_config["players"]
        self.extras = env_config.get("extras", [])

        n_actions = 2 * env_config["hand_size"] + (env_config["colors"] + env_config["ranks"]) * (
                env_config["players"] - 1)
        self.action_space = LegalActionDiscrete(n_actions, self)
        self.card_map = self.build_card_map(env_config)
        self.n_cards = env_config["colors"] * env_config["ranks"]
        self.observation_space = self.build_observation_space()

    def build_card_map(self, env_config):
        combinations = [str(x)+str(y) for x in self.state["player_observations"][0]["fireworks"].keys() for y in range(env_config["ranks"])]
        return {value: key for key, value in enumerate(combinations)}

    def build_observation_space(self):
        sample_obs = self.reset()[0]
        spaces = {"board": Box(
            low=sample_obs["board"].min(),
            high=sample_obs["board"].max(),
            shape=sample_obs["board"].shape,
            dtype=sample_obs["board"].dtype),
            "legal_actions": Box(
                low=sample_obs["legal_actions"].min(),
                high=sample_obs["legal_actions"].max(),
                shape=sample_obs["legal_actions"].shape,
                dtype=sample_obs["legal_actions"].dtype)}
        if "previous_round" in self.extras:
            spaces.update({"previous_round": Box(
                low=0,
                high=1,
                shape=sample_obs["previous_round"].shape,
                dtype=sample_obs["previous_round"].dtype)})
        if "hidden_hand" in self.extras:
            spaces.update({"hidden_hand": Box(
                low=sample_obs["hidden_hand"].min(),
                high=sample_obs["hidden_hand"].max(),
                shape=sample_obs["hidden_hand"].shape,
                dtype=sample_obs["hidden_hand"].dtype)})
        return Dict(spaces)

    def reset(self):
        self.state = self.env.reset()
        self.cum_reward[:] = 0
        self.last_action[:] = -1

        current_player, current_player_obs = self.extract_current_player_obs(self.state)
        return {current_player: current_player_obs}

    def step(self, action_dict):
        current_player = self.state["current_player"]
        current_player_action = action_dict[current_player]
        self.last_action[current_player] = current_player_action
        self.cum_reward[current_player] = 0
        assert self.action_space.contains(current_player_action)

        self.state, reward, done, _ = self.env.step(current_player_action.item())
        reward = reward / self.n_players  # scale the reward for each player - rllib sums up all player rewards
        self.cum_reward += reward
        next_player, next_player_obs = self.extract_current_player_obs(self.state)

        obs_dict = {next_player: next_player_obs}
        reward_dict = {next_player: self.cum_reward[next_player]}
        done_dict = {"__all__": done}
        if done:
            obs_dict = {player: next_player_obs for player in range(self.n_players)}
            reward_dict = {player: reward for player, reward in enumerate(self.cum_reward)}
        return obs_dict, reward_dict, done_dict, {}

    def extract_current_player_obs(self, all_obs):
        current_player = all_obs["current_player"]
        current_player_obs = numpy.array(all_obs["player_observations"][current_player]["vectorized"], dtype=numpy.int)
        legal_actions = self.legal_actions_as_int_to_bool(
            all_obs["player_observations"][current_player]["legal_moves_as_int"])
        obs = {
            "board": current_player_obs,
            "legal_actions": legal_actions,
        }
        obs = self.add_extras_to_obs(obs, current_player)
        return current_player, obs

    def add_extras_to_obs(self, obs, current_player):
        if "previous_round" in self.extras:
            obs.update({"previous_round": self.build_previous_round_actions(current_player)})
        if "hidden_hand" in self.extras:
            obs.update({"hidden_hand": self.build_hidden_hand(current_player)})
        return obs

    def legal_actions_as_int_to_bool(self, legal_moves_as_int):
        return (numpy.in1d(numpy.arange(self.action_space.n), numpy.array(legal_moves_as_int))).astype(numpy.int)

    def build_previous_round_actions(self, current_player):
        prev_round_int = numpy.roll(self.last_action, -current_player)[1:]
        prev_round_onehot = to_onehot(prev_round_int, self.action_space.n) / (self.n_players - 1)
        return prev_round_onehot.flatten()

    def build_hidden_hand(self, current_player):
        player_hand = self.state["player_observations"][(current_player + 1) % self.n_players]["observed_hands"][
            self.n_players - 1]
        cards_as_int = numpy.array([self.card_map[str(card["color"])+str(card["rank"])] for card in player_hand])
        padded_cards_as_int = numpy.full(self.env_config["hand_size"], -1)
        padded_cards_as_int[:cards_as_int.shape[0]] = cards_as_int
        return to_onehot(padded_cards_as_int, self.n_cards)


def env_creator(env_config):
    return MultiAgentHanabiEnv(env_config)


def to_onehot(indexes, n_classes):
    indexes = numpy.array(indexes)
    one_hots = numpy.zeros((indexes.size, n_classes), dtype=numpy.int)
    one_hots[numpy.arange(indexes.size), indexes] = indexes + 1
    one_hots[one_hots > 0] = 1
    one_hots[one_hots < 0] = 0
    return one_hots
