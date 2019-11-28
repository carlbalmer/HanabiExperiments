from ray import tune
from ray.rllib.models import ModelCatalog

from HanabiExperiments.environment import MultiAgentHanabiEnv, env_creator
from HanabiExperiments.model import IgnoreLegalActionsFCModel
from HanabiExperiments.preprocessor import OriginalSpaceSamplingDictFlatteningPreprocessor
from HanabiExperiments.policy import LegalActionDQNTrainer, LegalActionApexTrainer

ModelCatalog.register_custom_model("ILA_FC", IgnoreLegalActionsFCModel)
ModelCatalog.register_custom_preprocessor("OriginalSpaceSamplingPreprocessor",
                                          OriginalSpaceSamplingDictFlatteningPreprocessor)

tune.register_trainable("LegalActionDQN", LegalActionDQNTrainer)
tune.register_trainable("LegalActionApex", LegalActionApexTrainer)
tune.register_env("Hanabi", env_creator)
