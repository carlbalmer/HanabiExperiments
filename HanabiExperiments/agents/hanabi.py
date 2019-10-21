import torch
from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.agents.dqn.dqn_agent import DqnAgent, AgentInfo
from rlpyt.utils.buffer import buffer_to

from HanabiExperiments.distributions.epsilon_greedy import LegalActionEpsilonGreedy
from HanabiExperiments.envs.hanabi import HanabiMixin
from HanabiExperiments.models.hanabi import HanabiDuelDqnModel, HanabiCatDqnModel


class HanabiDuelDqnAgent(HanabiMixin, DqnAgent):

    def __init__(self, ModelCls=HanabiDuelDqnModel, **kwargs):
        super(HanabiDuelDqnAgent, self).__init__(ModelCls=ModelCls, **kwargs)

    def initialize(self, env_spaces, **kwargs):
        super(HanabiDuelDqnAgent, self).initialize(env_spaces, **kwargs)
        self.distribution = LegalActionEpsilonGreedy(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        q = self.model(*model_inputs)
        q = q.cpu()
        action = self.distribution.sample(q, observation.legal_actions)
        agent_info = AgentInfo(q=q)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)


class HanabiCatDqnAgent(HanabiMixin, CatDqnAgent):

    def __init__(self, ModelCls=HanabiCatDqnModel, **kwargs):
        super(HanabiCatDqnAgent, self).__init__(ModelCls=ModelCls, **kwargs)

