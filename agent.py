from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.samplers.buffer import get_example_outputs
from environment import CartPoleMixin
from model import CartPoleDqnModel


class CartPoleDQNAgent(CartPoleMixin, CatDqnAgent):

    def __init__(self, ModelCls=CartPoleDqnModel, **kwargs):
        super(CartPoleDQNAgent, self).__init__(ModelCls=ModelCls, **kwargs)