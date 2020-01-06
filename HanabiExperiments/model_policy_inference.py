import numpy
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.tf_ops import reduce_mean_ignore_inf
from ray.rllib.models.model import restore_original_dimensions

tf = try_import_tf()


class LegalActionsPolicyInferenceModel(DistributionalQModel, TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(LegalActionsPolicyInferenceModel, self).__init__(obs_space, action_space,
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

        self.policy_module = FullyConnectedNetwork(obs_module_output_dummy, None,
                                                   model_config["custom_options"]["policy_module_hiddens"][-1],
                                                   {"fcnet_activation": model_config["fcnet_activation"],
                                                    "fcnet_hiddens": model_config["custom_options"][
                                                        "policy_module_hiddens"],
                                                    "no_final_linear": True,
                                                    "vf_share_layers": True},
                                                   name + "policy_module")

        policy_head_input_dummy = numpy.zeros(
            action_space.n + model_config["custom_options"]["policy_module_hiddens"][-1])
        self.policy_head = FullyConnectedNetwork(policy_head_input_dummy, None,
                                                 obs_space.original_space["previous_round"].shape[0],
                                                 {"fcnet_activation": model_config["fcnet_activation"],
                                                  "fcnet_hiddens": model_config["custom_options"][
                                                      "policy_head_hiddens"],
                                                  "no_final_linear": False,
                                                  "vf_share_layers": True},
                                                 name + "policy_head")
        self.register_variables(self.obs_module.variables())
        self.register_variables(self.q_module.variables())
        self.register_variables(self.policy_module.variables())
        self.register_variables(self.policy_head.variables())
        self.q_config = {"num_atoms": kwargs["num_atoms"], "dueling": kwargs["dueling"]}
        self.q_out = None

    def forward(self, input_dict, state, seq_lens):
        obs_module_out, state_1 = self.obs_module({"obs": input_dict["obs"]["board"]}, state, seq_lens)
        q_module_out, state_2 = self.q_module({"obs": obs_module_out}, state_1, seq_lens)
        policy_module_out, state_3 = self.policy_module({"obs": obs_module_out}, state_1, seq_lens)

        model_out = tf.multiply(q_module_out, tf.stop_gradient(policy_module_out))

        self.calculate_and_store_q(input_dict, model_out)

        return model_out, state_2

    def calculate_and_store_q(self, input_dict, model_out):
        if self.q_config["num_atoms"] > 1:
            (action_scores, z, support_logits_per_action, logits,
             dist) = self.get_q_value_distributions(model_out)
        else:
            (action_scores, logits,
             dist) = self.get_q_value_distributions(model_out)
        if self.q_config["dueling"]:
            state_score = self.get_state_value(model_out)
            if self.q_config["num_atoms"] > 1:
                support_logits_per_action_mean = tf.reduce_mean(
                    support_logits_per_action, 1)
                support_logits_per_action_centered = (
                        support_logits_per_action - tf.expand_dims(support_logits_per_action_mean, 1))
                support_logits_per_action = tf.expand_dims(
                    state_score, 1) + support_logits_per_action_centered
                support_prob_per_action = tf.nn.softmax(
                    logits=support_logits_per_action)
                value = tf.reduce_sum(
                    input_tensor=z * support_prob_per_action, axis=-1)
                logits = support_logits_per_action
                dist = support_prob_per_action
            else:
                action_scores_mean = reduce_mean_ignore_inf(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(
                    action_scores_mean, 1)
                value = state_score + action_scores_centered
        else:
            value = action_scores
        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.cast(tf.maximum(tf.log(input_dict["obs"]["legal_actions"]), tf.float32.min), dtype=tf.float32)
        value = value + inf_mask
        self.q_out = {"value": value, "logits": logits, "dist": dist}

    def value_function(self):
        return self.fc.value_function()

    def get_q_out(self):
        temp = self.q_out
        self.q_out = None
        return temp

    def custom_loss(self, policy_loss, loss_inputs):
        obs = restore_original_dimensions(loss_inputs["obs"], self.obs_space, self.framework)["board"]
        previous_round = restore_original_dimensions(loss_inputs["new_obs"], self.obs_space, self.framework)["previous_round"]
        obs_module_out, state_1 = self.obs_module({"obs": obs}, None, None)
        policy_module_out, state_2 = self.policy_module({"obs": obs_module_out}, state_1, None)
        concat = tf.concat([tf.one_hot(tf.stop_gradient(loss_inputs["actions"]), self.action_space.n), policy_module_out], axis=1)
        policy_head_out, _ = self.policy_head({"obs": concat}, state_2, None)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(tf.stop_gradient(previous_round)), logits=policy_head_out)
        self.policy_inference_loss = tf.reduce_mean(cross_entropy)
        self.q_loss = policy_loss
        self.loss = (1 / tf.math.sqrt(self.policy_inference_loss)) * self.q_loss + self.policy_inference_loss
        return self.loss

    def metrics(self):
        return {
            "policy_inference_loss": self.policy_inference_loss,
            "q_loss": self.q_loss,
            "loss": self.loss
        }