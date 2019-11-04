from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


class IgnoreLegalActionsFCModel(DistributionalQModel, TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(IgnoreLegalActionsFCModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)
        self.fc = FullyConnectedNetwork(obs_space.original_space["board"], action_space, num_outputs, model_config, name + "fc")
        self.register_variables(self.fc.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.fc({"obs": input_dict["obs"]["board"]}, state, seq_lens)

    def value_function(self):
        return self.fc.value_function()