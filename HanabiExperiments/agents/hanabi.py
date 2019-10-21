import torch
from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent, AgentInfo as CatAgentInfo
from rlpyt.agents.dqn.dqn_agent import DqnAgent, AgentInfo as DqnAgentInfo
from rlpyt.utils.buffer import buffer_to

from HanabiExperiments.distributions.epsilon_greedy import CategorialLegalActionEpsilonGreedy, LegalActionEpsilonGreedy
from HanabiExperiments.envs.hanabi import HanabiMixin
from HanabiExperiments.models.hanabi import HanabiDuelDqnModel, HanabiCatDqnModel


class LegalActionAgentMixin:

    def initialize(self, env_spaces, **kwargs):
        super(LegalActionAgentMixin, self).initialize(env_spaces, **kwargs)
        if isinstance(self, CatDqnAgent):
            self.distribution = CategorialLegalActionEpsilonGreedy(dim=env_spaces.action.n,
                                                                   z=torch.linspace(-1, 1, self.n_atoms))
        else:
            self.distribution = LegalActionEpsilonGreedy(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        model_output = self.model(*model_inputs)
        model_output = model_output.cpu()
        _, legal_actions = torch.split(observation, (self.env_spaces.observation.shape[0], self.env_spaces.action.n),
                                       dim=-1)
        action = self.distribution.sample(model_output, legal_actions)
        if isinstance(self, CatDqnAgent):
            agent_info = CatAgentInfo(p=model_output)
        else:
            agent_info = DqnAgentInfo(q=model_output)
        return AgentStep(action=action, agent_info=agent_info)


class HanabiDuelDqnAgent(HanabiMixin, LegalActionAgentMixin, DqnAgent):

    def __init__(self, ModelCls=HanabiDuelDqnModel, **kwargs):
        super(HanabiDuelDqnAgent, self).__init__(ModelCls=ModelCls, **kwargs)


class HanabiCatDqnAgent(HanabiMixin, LegalActionAgentMixin, CatDqnAgent):

    def __init__(self, ModelCls=HanabiCatDqnModel, **kwargs):
        super(HanabiCatDqnAgent, self).__init__(ModelCls=ModelCls, **kwargs)
