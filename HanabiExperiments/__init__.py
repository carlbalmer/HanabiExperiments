from ray import tune
from ray.rllib.models import ModelCatalog

from HanabiExperiments.environment import MultiAgentHanabiEnv, env_creator
from HanabiExperiments.models.fc import HanabiFullyConnected
from HanabiExperiments.models.hand_inference import HanabiHandInference, HanabiHandInferenceIndependentLoss
from HanabiExperiments.models.policy_inference import HanabiPolicyInference, HanabiTargetNNPolicyInference, \
    HanabiPolicyInferenceIndependentLoss, HanabiTargetNNPolicyInferenceIndependentLoss
from HanabiExperiments.policies.target_network_policy_inference import TargetPolicyInferenceApexTrainer, \
    TargetPolicyInferenceDQNTrainer
from HanabiExperiments.preprocessor import OriginalSpaceSamplingDictFlatteningPreprocessor
from HanabiExperiments.policies.legal_action import LegalActionDQNTrainer, LegalActionApexTrainer

ModelCatalog.register_custom_model("Hanabi_FC", HanabiFullyConnected)
ModelCatalog.register_custom_model("Hanabi_PolicyInference", HanabiPolicyInference)
ModelCatalog.register_custom_model("Hanabi_HandInference", HanabiHandInference)
ModelCatalog.register_custom_model("Hanabi_TargetPolicyInference", HanabiTargetNNPolicyInference)
ModelCatalog.register_custom_model("Hanabi_PolicyInference_independent_loss", HanabiPolicyInferenceIndependentLoss)
ModelCatalog.register_custom_model("Hanabi_TargetPolicyInference_independent_loss", HanabiTargetNNPolicyInferenceIndependentLoss)
ModelCatalog.register_custom_model("Hanabi_HandInference_independent_loss", HanabiHandInferenceIndependentLoss)

ModelCatalog.register_custom_preprocessor("OriginalSpaceSamplingPreprocessor",
                                          OriginalSpaceSamplingDictFlatteningPreprocessor)

tune.register_trainable("LegalActionDQN", LegalActionDQNTrainer)
tune.register_trainable("LegalActionApex", LegalActionApexTrainer)
tune.register_trainable("TargetPolicyInferenceDQN",TargetPolicyInferenceDQNTrainer)
tune.register_trainable("TargetPolicyInferenceApex",TargetPolicyInferenceApexTrainer)

tune.register_env("Hanabi", env_creator)
