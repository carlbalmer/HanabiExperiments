from rlpyt.agents.dqn.dqn_agent import DqnAgent

from environment import CartPoleMixin
from model import CartPoleDqnModel


class CartPoleDQNAgent(CartPoleMixin, DqnAgent):

    def __init__(self, ModelCls=CartPoleDqnModel, **kwargs):
        super(CartPoleDQNAgent, self).__init__(ModelCls=ModelCls, **kwargs)