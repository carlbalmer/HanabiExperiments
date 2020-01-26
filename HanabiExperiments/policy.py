from ray.rllib import SampleBatch
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG, APEX_TRAINER_PROPERTIES
from ray.rllib.agents.dqn.dqn_policy import _build_parameter_noise, QValuePolicy, QLoss, PRIO_WEIGHTS, DQNTFPolicy
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


def build_q_networks(policy, q_model, input_dict, obs_space, action_space,
                     config):

    # Action Q network
    q_values, q_logits, q_dist = _compute_q_values(
        policy, q_model, input_dict[SampleBatch.CUR_OBS], obs_space,
        action_space)
    policy.q_values = q_values
    policy.q_func_vars = q_model.variables()

    # Noise vars for Q network except for layer normalization vars
    if config["parameter_noise"]:
        _build_parameter_noise(
            policy,
            [var for var in policy.q_func_vars if "LayerNorm" not in var.name])
        policy.action_probs = tf.nn.softmax(policy.q_values)

    # Action outputs
    qvp = QValuePolicy(q_values, input_dict[SampleBatch.CUR_OBS],
                       action_space.n, policy.cur_epsilon, config["soft_q"],
                       config["softmax_temp"], config["model"])
    policy.output_actions, policy.action_prob = qvp.action, qvp.action_prob

    actions = policy.output_actions
    action_prob = (tf.log(policy.action_prob)
                   if policy.action_prob is not None else None)
    return actions, action_prob


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
        train_batch[PRIO_WEIGHTS], train_batch[SampleBatch.REWARDS],
        tf.cast(train_batch[SampleBatch.DONES],
                tf.float32), config["gamma"], config["n_step"],
        config["num_atoms"], config["v_min"], config["v_max"])

    policy.q_loss.stats.update({
        "q_loss": policy.q_loss.loss
    })

    loss = policy.q_model.extra_loss(policy.q_loss.loss, train_batch, policy.q_loss.stats)

    return loss


def _compute_q_values(policy, model, obs, obs_space, action_space):
    model({
        "obs": obs,
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    q_out = model.get_q_out()

    return q_out["value"], q_out["logits"], q_out["dist"]


LegalActionDQNPolicy = DQNTFPolicy.with_updates(
    name="LegalActionDQNPolicy",
    action_sampler_fn=build_q_networks,
    loss_fn=build_q_losses)

LegalActionDQNTrainer = DQNTrainer.with_updates(
    name="LegalActionDQN",
    default_policy=LegalActionDQNPolicy)

LegalActionApexTrainer = LegalActionDQNTrainer.with_updates(
    name="LegalActionAPEX", default_config=APEX_DEFAULT_CONFIG, **APEX_TRAINER_PROPERTIES)