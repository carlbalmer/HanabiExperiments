import ray
from ray import tune
from ray.rllib.models import ModelCatalog

from rayExperiments.environment import MultiAgentHanabiEnv, HANABI_CONF_FULL_4p
from rayExperiments.model import IgnoreLegalActionsFCModel
from rayExperiments.policy import LegalActionDQNTrainer
from rayExperiments.preprocessor import OriginalSpaceSamplingDictFlatteningPreprocessor

ModelCatalog.register_custom_model("ILA_FC", IgnoreLegalActionsFCModel)
ModelCatalog.register_custom_preprocessor("OriginalSpaceSamplingPreprocessor",
                                          OriginalSpaceSamplingDictFlatteningPreprocessor)

tune.run(LegalActionDQNTrainer, config={
    "env": MultiAgentHanabiEnv,
    "env_config": HANABI_CONF_FULL_4p,
    "model": {
        "custom_model": "ILA_FC",
        "custom_preprocessor": "OriginalSpaceSamplingPreprocessor",
            }
})
