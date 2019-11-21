import numpy
from gym.spaces import Discrete, Box
from ray.rllib.models.preprocessors import get_preprocessor


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


class OriginalSpaceSamplingBox(Box):

    def __init__(self, original_space, low, high, shape=None, dtype=numpy.float32):
        super(OriginalSpaceSamplingBox, self).__init__(low, high, shape=shape, dtype=dtype)
        self.original_space = original_space

    def sample(self):
        sample = dict(self.original_space.sample())
        prep = get_preprocessor(self.original_space)(self.original_space)
        return prep.transform(sample)
