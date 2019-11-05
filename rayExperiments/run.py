from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy
from ray.rllib.models import ModelCatalog

from rayExperiments.environment import MultiAgentHanabiEnv, HANABI_CONF_FULL_4p
from rayExperiments.model import IgnoreLegalActionsFCModel
from rayExperiments.policy import build_q_networks, build_q_losses
from rayExperiments.preprocessor import OriginalSpaceSamplingDictFlatteningPreprocessor

ModelCatalog.register_custom_model("ILA_FC", IgnoreLegalActionsFCModel)
ModelCatalog.register_custom_preprocessor("OriginalSpaceSamplingPreprocessor", OriginalSpaceSamplingDictFlatteningPreprocessor)

LegalActionDQNPolicy = DQNTFPolicy.with_updates(
    name="LegalActionDQNPolicy",
    action_sampler_fn=build_q_networks,
    loss_fn=build_q_losses)

LegalActionDQNTrainer = DQNTrainer.with_updates(
    default_policy=LegalActionDQNPolicy)

tune.run(LegalActionDQNTrainer, config={
    "env": MultiAgentHanabiEnv,
    "env_config": HANABI_CONF_FULL_4p,
    "num_workers": 3,
    "num_gpus": 0.25,
    "num_gpus_per_worker": 0.25,
    "num_envs_per_worker": 20,
    "model": {
        "custom_model": "ILA_FC",
        "custom_preprocessor": "OriginalSpaceSamplingPreprocessor",
            },
})