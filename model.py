from rlpyt.models.mlp import MlpModel


class CartPoleDqnModel(MlpModel):
    def forward(self, observation, prev_action, prev_reward):
        return super(CartPoleDqnModel, self).forward(observation.float())