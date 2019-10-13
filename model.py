from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel


class CartPoleDqnModel(DistributionalDuelingHeadModel):

    def __init__(self, **kwargs):
        super(CartPoleDqnModel, self).__init__(**kwargs)
        self.first = True

    def forward(self, observation, prev_action, prev_reward):
        if self.first == True:
            self.first = False
            return super(CartPoleDqnModel, self).forward(observation.float()).squeeze()
        return super(CartPoleDqnModel, self).forward(observation.float())