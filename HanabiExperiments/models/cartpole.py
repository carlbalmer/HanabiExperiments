from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel


class CartPoleDqnModel(DistributionalDuelingHeadModel):

    def forward(self, observation, prev_action, prev_reward):
        return super(CartPoleDqnModel, self).forward(observation.float())