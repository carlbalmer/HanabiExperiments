import torch
from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.agents.dqn.dqn_agent import DqnAgent, AgentInfo
from rlpyt.utils.buffer import buffer_to

from HanabiExperiments.envs.hanabi import HanabiMixin
from HanabiExperiments.models.hanabi import HanabiDuelDqnModel, HanabiCatDqnModel


class HanabiDuelDqnAgent(HanabiMixin, DqnAgent):

    def __init__(self, ModelCls=HanabiDuelDqnModel, **kwargs):
        super(HanabiDuelDqnAgent, self).__init__(ModelCls=ModelCls, **kwargs)


class HanabiCatDqnAgent(HanabiMixin, CatDqnAgent):

    def __init__(self, ModelCls=HanabiCatDqnModel, **kwargs):
        super(HanabiCatDqnAgent, self).__init__(ModelCls=ModelCls, **kwargs)

