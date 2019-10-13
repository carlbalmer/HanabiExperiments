from rlpyt.models.dqn.dueling import DuelingHeadModel


class CartPoleDqnModel(DuelingHeadModel):
    def forward(self, observation, prev_action, prev_reward):
        return super(CartPoleDqnModel, self).forward(observation.float())