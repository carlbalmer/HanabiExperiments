from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from HanabiExperiments.envs.cartpole import CartPoleMixin
from HanabiExperiments.models.cartpole import CartPoleDqnModel


class CartPoleDQNAgent(CartPoleMixin, CatDqnAgent):

    def __init__(self, ModelCls=CartPoleDqnModel, **kwargs):
        super(CartPoleDQNAgent, self).__init__(ModelCls=ModelCls, **kwargs)