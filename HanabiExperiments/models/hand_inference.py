import numpy
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf

from HanabiExperiments.models.base import LegalActionsDistributionalQModel, get_aux_loss_formula

tf = try_import_tf()


class HanabiHandInference(LegalActionsDistributionalQModel):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(HanabiHandInference, self).__init__(obs_space, action_space,
                                                         model_config["custom_options"]["q_module_hiddens"][-1],
                                                         model_config, name,
                                                         **kwargs)
        self.obs_module = FullyConnectedNetwork(obs_space.original_space["board"],
                                                None,
                                                model_config["custom_options"]["obs_module_hiddens"][-1],
                                                {
                                                    "fcnet_activation": model_config["fcnet_activation"],
                                                    "fcnet_hiddens": model_config["custom_options"][
                                                        "obs_module_hiddens"],
                                                    "no_final_linear": True,
                                                    "vf_share_layers": True},
                                                name + "obs_module")

        obs_module_output_dummy = numpy.zeros(model_config["custom_options"]["obs_module_hiddens"][-1])
        self.q_module = FullyConnectedNetwork(obs_module_output_dummy, None,
                                              model_config["custom_options"]["q_module_hiddens"][-1],
                                              {"fcnet_activation": model_config["fcnet_activation"],
                                               "fcnet_hiddens": model_config["custom_options"]["q_module_hiddens"],
                                               "no_final_linear": True,
                                               "vf_share_layers": True},
                                              name + "q_module")

        self.aux_module = FullyConnectedNetwork(obs_module_output_dummy, None,
                                                model_config["custom_options"]["aux_module_hiddens"][-1],
                                                {"fcnet_activation": model_config["fcnet_activation"],
                                                    "fcnet_hiddens": model_config["custom_options"][
                                                        "aux_module_hiddens"],
                                                    "no_final_linear": True,
                                                    "vf_share_layers": True},
                                                name + "aux_module")

        aux_head_input_dummy = numpy.zeros(model_config["custom_options"]["aux_module_hiddens"][-1])
        self.aux_head = FullyConnectedNetwork(aux_head_input_dummy, None,
                                              numpy.prod(obs_space.original_space["hidden_hand"].shape),
                                              {"fcnet_activation": model_config["fcnet_activation"],
                                                  "fcnet_hiddens": model_config["custom_options"][
                                                      "aux_head_hiddens"],
                                                  "no_final_linear": False,
                                                  "vf_share_layers": True},
                                              name + "aux_head")
        self.register_variables(self.obs_module.variables())
        self.register_variables(self.q_module.variables())
        self.register_variables(self.aux_module.variables())
        self.register_variables(self.aux_head.variables())
        self.aux_loss_formula = get_aux_loss_formula(model_config["custom_options"].get("aux_loss_formula", "sqrt"))

    def forward(self, input_dict, state, seq_lens):
        obs_module_out, state_1 = self.obs_module({"obs": input_dict["obs"]["board"]}, state, seq_lens)
        q_module_out, state_2 = self.q_module({"obs": obs_module_out}, state_1, seq_lens)
        aux_module_out, state_3 = self.aux_module({"obs": obs_module_out}, state_1, seq_lens)

        model_out = tf.multiply(q_module_out, tf.stop_gradient(aux_module_out))

        self.calculate_and_store_q(input_dict, model_out)

        return model_out, state_2

    def extra_loss(self, policy_loss, loss_inputs, stats):
        obs = restore_original_dimensions(loss_inputs["obs"], self.obs_space, self.framework)["board"]
        hidden_hand = restore_original_dimensions(loss_inputs["obs"], self.obs_space, self.framework)[
            "hidden_hand"]
        hidden_hand = tf.reshape(hidden_hand,[tf.shape(hidden_hand)[0],hidden_hand.shape[1]*hidden_hand.shape[2]])  # reshape so all hands are in one vector
        hidden_hand = tf.math.divide_no_nan(hidden_hand, tf.expand_dims(tf.reduce_sum(hidden_hand, 1), 1))  # normalize so sum of vector is 1
        obs_module_out, state_1 = self.obs_module({"obs": obs}, None, None)
        aux_module_out, state_2 = self.aux_module({"obs": obs_module_out}, state_1, None)
        aux_head_out, _ = self.aux_head({"obs": aux_module_out}, state_2, None)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.stop_gradient(hidden_hand),
            logits=aux_head_out)
        hand_inference_loss = tf.reduce_mean(cross_entropy)
        combined_loss = self.aux_loss_formula(policy_loss, hand_inference_loss)
        stats.update({
            "combined_loss": combined_loss,
            "hand_inference_loss": hand_inference_loss
        })
        return combined_loss


class HanabiHandInferenceIndependentLoss(HanabiHandInference):

    def extra_loss(self, policy_loss, loss_inputs, stats):
        obs = restore_original_dimensions(loss_inputs["obs"], self.obs_space, self.framework)["board"]
        hidden_hand = restore_original_dimensions(loss_inputs["obs"], self.obs_space, self.framework)[
            "hidden_hand"]
        hidden_hand = tf.reshape(hidden_hand,[tf.shape(hidden_hand)[0] * hidden_hand.shape[1], hidden_hand.shape[2]])  # reshape so all hands are in one batch
        obs_module_out, state_1 = self.obs_module({"obs": obs}, None, None)
        aux_module_out, state_2 = self.aux_module({"obs": obs_module_out}, state_1, None)
        aux_head_out, _ = self.aux_head({"obs": aux_module_out}, state_2, None)
        aux_head_out = tf.reshape(aux_head_out, tf.shape(hidden_hand))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.stop_gradient(hidden_hand),
            logits=aux_head_out)
        hand_inference_loss = tf.reduce_mean(cross_entropy)
        combined_loss = self.aux_loss_formula(policy_loss, hand_inference_loss)
        stats.update({
            "combined_loss": combined_loss,
            "hand_inference_loss": hand_inference_loss
        })
        return combined_loss
