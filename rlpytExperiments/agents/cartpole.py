from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpytExperiments.envs.cartpole import CartPoleMixin
from rlpytExperiments.models.cartpole import CartPoleDqnModel


class CartPoleDQNAgent(CartPoleMixin, CatDqnAgent):

    def __init__(self, ModelCls=CartPoleDqnModel, **kwargs):
        super(CartPoleDQNAgent, self).__init__(ModelCls=ModelCls, **kwargs)