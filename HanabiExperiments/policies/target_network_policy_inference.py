from ray.rllib import SampleBatch
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG, APEX_TRAINER_PROPERTIES
from ray.rllib.agents.dqn.dqn_policy import QLoss, PRIO_WEIGHTS
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.utils import try_import_tf

from HanabiExperiments.policies.legal_action import LegalActionDQNPolicy, _compute_q_values

tf = try_import_tf()


def build_q_losses(policy, model, _, train_batch):
    config = policy.config
    # q network evaluation
    q_t, q_logits_t, q_dist_t = _compute_q_values(
        policy, policy.q_model, train_batch[SampleBatch.CUR_OBS],
        policy.observation_space, policy.action_space)

    # target q network evalution
    q_tp1, q_logits_tp1, q_dist_tp1 = _compute_q_values(
        policy, policy.target_q_model, train_batch[SampleBatch.NEXT_OBS],
        policy.observation_space, policy.action_space)
    policy.target_q_func_vars = policy.target_q_model.variables()

    # q scores for actions which we know were selected in the given state.
    one_hot_selection = tf.one_hot(
        tf.cast(train_batch[SampleBatch.ACTIONS], tf.int32),
        policy.action_space.n)
    q_t_selected = tf.reduce_sum(q_t * one_hot_selection, 1)
    q_logits_t_selected = tf.reduce_sum(
        q_logits_t * tf.expand_dims(one_hot_selection, -1), 1)

    # compute estimate of best possible value starting from state at t + 1
    if config["double_q"]:
        q_tp1_using_online_net, q_logits_tp1_using_online_net, \
            q_dist_tp1_using_online_net = _compute_q_values(
                policy, policy.q_model,
                train_batch[SampleBatch.NEXT_OBS],
                policy.observation_space, policy.action_space)
        q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
        q_tp1_best_one_hot_selection = tf.one_hot(q_tp1_best_using_online_net,
                                                  policy.action_space.n)
        q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        q_dist_tp1_best = tf.reduce_sum(
            q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1)
    else:
        q_tp1_best_one_hot_selection = tf.one_hot(
            tf.argmax(q_tp1, 1), policy.action_space.n)
        q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        q_dist_tp1_best = tf.reduce_sum(
            q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1)

    policy.q_loss = QLoss(
        q_t_selected, q_logits_t_selected, q_tp1_best, q_dist_tp1_best,
        tf.cast(train_batch[PRIO_WEIGHTS], tf.float32), train_batch[SampleBatch.REWARDS],
        tf.cast(train_batch[SampleBatch.DONES],
                tf.float32), config["gamma"], config["n_step"],
        config["num_atoms"], config["v_min"], config["v_max"])

    policy.q_loss.stats.update({
        "q_loss": policy.q_loss.loss
    })

    targets = get_actions_from_target_net(train_batch, policy, policy.target_q_model, policy.observation_space, policy.action_space)
    loss = policy.q_model.extra_loss(policy.q_loss.loss, train_batch, targets, policy.q_loss.stats)

    return loss


def get_actions_from_target_net(train_batch, policy, target_q_model, observation_space, action_space):
    restored =restore_original_dimensions(train_batch[SampleBatch.NEXT_OBS], observation_space, target_q_model.framework)
    previous_round_obs = {}
    previous_round_obs["board"] = tf.reshape(restored["previous_round"],
                                   [tf.shape(restored["previous_round"])[0]*
                                    restored["previous_round"].shape[1],
                                    restored["previous_round"].shape[2]])
    previous_round_obs["legal_actions"] = tf.reshape(restored["previous_round_legal_actions"],
                                           [tf.shape(restored["previous_round_legal_actions"])[0]*
                                            restored["previous_round_legal_actions"].shape[1],
                                            restored["previous_round_legal_actions"].shape[2]])
    target_q_model.forward({"obs":previous_round_obs, "is_training": policy._get_is_training_placeholder()}, [], None)
    q_out = target_q_model.get_q_out()
    previous_round = tf.one_hot(
        tf.argmax(q_out["value"], 1), policy.action_space.n)
    previous_round = tf.reshape(previous_round,
                                [tf.shape(restored["previous_round"])[0],
                                 restored["previous_round"].shape[1],
                                 action_space.n])
    return previous_round


TargetPolicyInferenceDQNPolicy = LegalActionDQNPolicy.with_updates(
    name="TargetPolicyInferenceDQNPolicy",
    loss_fn=build_q_losses)

TargetPolicyInferenceDQNTrainer = DQNTrainer.with_updates(
    name="TargetPolicyInferenceDQN",
    default_policy=TargetPolicyInferenceDQNPolicy)

TargetPolicyInferenceApexTrainer = TargetPolicyInferenceDQNTrainer.with_updates(
    name="TargetPolicyInferenceAPEX", default_config=APEX_DEFAULT_CONFIG, **APEX_TRAINER_PROPERTIES)
