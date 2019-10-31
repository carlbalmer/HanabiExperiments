from ray import tune
from ray.rllib.agents.dqn import DQNTrainer

from rayExperiments.environment import MultiAgentHanabiEnv, HANABI_CONF_FULL_4p

tune.run(DQNTrainer, config={
    "env": MultiAgentHanabiEnv,
    "env_config": HANABI_CONF_FULL_4p
})