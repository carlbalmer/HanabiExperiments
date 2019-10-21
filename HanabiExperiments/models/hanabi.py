from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel, DuelingHeadModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class HanabiDuelDqnModel(DuelingHeadModel):

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, obs_shape = infer_leading_dims(observation.vectorized_obs, 1)
        p = super(HanabiDuelDqnModel, self).forward(
            observation.vectorized_obs.view(T * B, *obs_shape).float())
        p = p * observation.legal_actions.view(T * B, *observation.legal_actions.shape[-1:]).float()
        p = restore_leading_dims(p, lead_dim, T, B)
        return p


class HanabiCatDqnModel(DistributionalDuelingHeadModel):

    def forward(self, observation, prev_action, prev_reward):
        return super(HanabiCatDqnModel, self).forward(observation.float())