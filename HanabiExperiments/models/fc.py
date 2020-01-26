from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf

from HanabiExperiments.models.base import LegalActionsDistributionalQModel

tf = try_import_tf()


class HanabiFullyConnected(LegalActionsDistributionalQModel):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(HanabiFullyConnected, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)
        self.fc = FullyConnectedNetwork(obs_space.original_space["board"], action_space, num_outputs, model_config,
                                        name + "fc")
        self.register_variables(self.fc.variables())

    def forward(self, input_dict, state, seq_lens):
        model_out, state = self.fc({"obs": input_dict["obs"]["board"]}, state, seq_lens)
        self.calculate_and_store_q(input_dict, model_out)
        return model_out, state
