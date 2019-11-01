from ray import tune
from ray.rllib.agents.dqn import DQNTrainer

from rayExperiments.environment import MultiAgentHanabiEnv, HANABI_CONF_FULL_4p

tune.run(DQNTrainer, config={
    "env": MultiAgentHanabiEnv,
    "env_config": HANABI_CONF_FULL_4p,
    "num_workers": 3,
    "num_gpus": 1,
    "num_gpus_per_worker": 0,
    "num_envs_per_worker": 20
})