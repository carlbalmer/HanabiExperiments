from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.tf_ops import reduce_mean_ignore_inf

tf = try_import_tf()


class LegalActionsDistributionalQModel(DistributionalQModel):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(LegalActionsDistributionalQModel, self).__init__(obs_space, action_space, num_outputs, model_config, name,
                                                        **kwargs)
        self.q_config = {"num_atoms": kwargs["num_atoms"], "dueling": kwargs["dueling"]}
        self.q_out = None

    def forward(self, input_dict, state, seq_lens):
        raise NotImplementedError

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

    def extra_loss(self, policy_loss, loss_inputs, stats):
        return policy_loss