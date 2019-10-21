import torch
from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel, DuelingHeadModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class HanabiDuelDqnModel(DuelingHeadModel):

    def __init__(self, **kwargs):
        super(HanabiDuelDqnModel, self).__init__(**kwargs)
        self.input_size = kwargs['input_size']
        self.output_size = kwargs['output_size']

    def forward(self, observation, prev_action, prev_reward):
        # add batch dimension if not present & split into observation and legal moves
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)
        obs, legal_actions = torch.split(observation.view(T * B, *obs_shape), (self.input_size, self.output_size), dim=-1)

        p = super(HanabiDuelDqnModel, self).forward(obs.float())
        p[~legal_actions.bool()] = 0 #-float('inf')
        p = restore_leading_dims(p, lead_dim, T, B)
        return p


class HanabiCatDqnModel(DistributionalDuelingHeadModel):

    def forward(self, observation, prev_action, prev_reward):
        return super(HanabiCatDqnModel, self).forward(observation.float())