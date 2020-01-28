from ray import tune
from ray.rllib.models import ModelCatalog

from HanabiExperiments.environment import MultiAgentHanabiEnv, env_creator
from HanabiExperiments.models.fc import HanabiFullyConnected
from HanabiExperiments.models.hand_inference import HanabiHandInference
from HanabiExperiments.models.policy_inference import HanabiPolicyInference
from HanabiExperiments.preprocessor import OriginalSpaceSamplingDictFlatteningPreprocessor
from HanabiExperiments.policy import LegalActionDQNTrainer, LegalActionApexTrainer

ModelCatalog.register_custom_model("Hanabi_FC", HanabiFullyConnected)
ModelCatalog.register_custom_model("Hanabi_PolicyInference", HanabiPolicyInference)
ModelCatalog.register_custom_model("Hanabi_HandInference", HanabiHandInference)
ModelCatalog.register_custom_preprocessor("OriginalSpaceSamplingPreprocessor",
                                          OriginalSpaceSamplingDictFlatteningPreprocessor)

tune.register_trainable("LegalActionDQN", LegalActionDQNTrainer)
tune.register_trainable("LegalActionApex", LegalActionApexTrainer)
tune.register_env("Hanabi", env_creator)
