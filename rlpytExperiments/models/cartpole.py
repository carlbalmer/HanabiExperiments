from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class CartPoleDqnModel(DistributionalDuelingHeadModel):

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)
        p = super(CartPoleDqnModel, self).forward(observation.view(T * B, *obs_shape).float())
        p = restore_leading_dims(p, lead_dim, T, B)
        return p