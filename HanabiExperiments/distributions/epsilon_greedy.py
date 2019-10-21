import numpy
import torch
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class LegalActionEpsilonGreedy(EpsilonGreedy):

    def sample(self, q, legal_actions):
        # add batch dimension if it is not already there
        lead_dim, T, B, obs_shape = infer_leading_dims(q, 1)
        q_view = q.view(T * B, * obs_shape)
        # select action for each set of q values
        epsion_mask = torch.rand(q_view.shape[0]) < self._epsilon
        rand_q = torch.rand(q_view.shape)
        rand_q[~epsion_mask] = q_view[~epsion_mask]
        rand_q[~legal_actions.view(T * B, * obs_shape[-1:]).bool()] = -float('inf')
        arg_select = torch.argmax(rand_q, dim=-1)
        # remove batch dimension if it was not there in the beginning
        arg_select = restore_leading_dims(arg_select, lead_dim, T, B)
        return arg_select
