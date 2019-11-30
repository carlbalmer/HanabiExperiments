import numpy
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
from ray.rllib.utils.annotations import PublicAPI

from HanabiExperiments.spaces import OriginalSpaceSamplingBox


class OriginalSpaceSamplingPreprocessorMixin:

    @property
    @PublicAPI
    def observation_space(self):
        return OriginalSpaceSamplingBox(self._obs_space, -1., 1., self.shape, dtype=numpy.float32)


class OriginalSpaceSamplingDictFlatteningPreprocessor(OriginalSpaceSamplingPreprocessorMixin,
                                                      DictFlatteningPreprocessor):
    pass
