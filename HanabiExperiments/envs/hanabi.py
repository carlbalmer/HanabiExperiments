from collections import namedtuple

from rlpyt.envs.base import Env, EnvStep
from hanabi_learning_environment.rl_env import HanabiEnv
from rlpyt.spaces.int_box import IntBox
import numpy
from rlpyt.utils.collections import namedarraytuple

Observation = namedarraytuple("Observation", ["vectorized_obs", "legal_actions"])

class WrappedHanabiEnv(Env):

    def __init__(
            self,
            colors=5,
            ranks=5,
            players=4,
            hand_size=4,
            max_information_tokens=8,
            max_life_tokens=3,
            observation_type=1,
            seed=None,
            random_start_player=False
    ):
        r"""Creates an environment with the given game configuration.

        Args:
          colors: int, Number of colors \in [2,5].
          ranks: int, Number of ranks \in [2,5].
          players: int, Number of players \in [2,5].
          hand_size: int, Hand size \in [4,5].
          max_information_tokens: int, Number of information tokens (>=0).
          max_life_tokens: int, Number of life tokens (>=1).
          observation_type: int.
            0: Minimal observation.
            1: First-order common knowledge observation.
          seed: int, Random seed.
          random_start_player: bool, Random start player.
        """
        config = {
            "colors": colors,
            "ranks": ranks,
            "players": players,
            "hand_size": hand_size,
            "max_information_tokens": max_information_tokens,
            "max_life_tokens": max_life_tokens,
            "observation_type": observation_type,
            "random_start_player": random_start_player
        }
        if seed:
            config["seed"] = seed
        self.env = HanabiEnv(config)

        n_actions = 2 * hand_size + (colors + ranks) * (players - 1)
        self._action_space = IntBox(low=0, high=n_actions, dtype="int16")

        sample_obs = numpy.array(HanabiEnv(config).reset()["player_observations"][0]["vectorized"])
        self._observation_space = IntBox(low=0, high=2, shape=sample_obs.shape, dtype=sample_obs.dtype)

    def step(self, action):
        all_obs, reward, done, _ = self.env.step(action.item())
        return EnvStep(self.extract_current_player_obs(all_obs), reward, done, None)

    def reset(self):
        return self.extract_current_player_obs(self.env.reset())

    def legal_moves_as_int_to_bool(self, legal_moves_as_int):
        return numpy.in1d(numpy.arange(self.action_space.n), numpy.array(legal_moves_as_int))

    def extract_current_player_obs(self, all_observations):
        current_player_obs = all_observations["player_observations"][all_observations["current_player"]]
        legal_moves = self.legal_moves_as_int_to_bool(current_player_obs["legal_moves_as_int"])
        vectorized_obs = numpy.array(current_player_obs["vectorized"])
        return Observation(vectorized_obs, legal_moves)

    @property
    def horizon(self):
        pass


class HanabiMixin:

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(input_size=env_spaces.observation.shape[0],
                    output_size=env_spaces.action.n)

