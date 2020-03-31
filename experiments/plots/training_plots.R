# Title     : Training Plots
# Objective : Plot training progress for thesis
# Created by: carl
# Created on: 22.03.20
setwd("/home/carl/PycharmProjects/HanabiExperiments")

library("tidyverse")
cols = cols(experiment_tag = col_character())
cbp2 <- c("#000000", "#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442")
levels <- c("Baseline", "Ape-X DQN", "1%", "5%", "20%", "60%", "Adaptive", "1 round", "3 rounds", "6 rounds", "1 round deeper", "1 step", "3 step", "1 step target", "3 step target")

baseline = read_csv("experiments/runs/dqn/local-dqn-deepmind/LegalActionDQN_Hanabi_a47106c6_2020-01-26_21-15-48bkgpc4fw/progress.csv",col_names = T, cols) %>% add_column(name =  "Baseline")

stack_round_1 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/1_turn/progress.csv") %>% add_column(name =  "1 round", type = "Round Stacking")
stack_round_3 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/3_turn/progress.csv") %>% add_column(name =  "3 rounds", type = "Round Stacking")
stack_round_6 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/6_turn/progress.csv") %>% add_column(name =  "6 rounds", type = "Round Stacking")
stack_round_1_deeper = read_csv("experiments/runs/dqn/local-dqn-turn-stacking-deeper/LegalActionDQN_Hanabi_0f119e68_2020-03-14_11-46-28hpd71rwd/progress.csv", col_names = T, cols) %>% add_column(name =  "1 round deeper", type = "Round Stacking")

hand_adaptive = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference/independant_loss/progress.csv") %>% add_column(name =  "Adaptive", type = "Hand inference")
hand_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/1/progress.csv") %>% add_column(name =  "1%", type = "Hand inference")
hand_5 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/5/progress.csv") %>% add_column(name =  "5%", type = "Hand inference")
hand_20 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/20/progress.csv") %>% add_column(name =  "20%", type = "Hand inference")
hand_60 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/60/progress.csv") %>% add_column(name =  "60%", type = "Hand inference")

policy_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference/independant_loss_1_step/progress.csv") %>% add_column(name =  "1 step", type = "Policy inference")
policy_3 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference/independant_loss_3_step/progress.csv") %>% add_column(name =  "3 step", type = "Policy inference")
target_policy_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-target/independant_loss_1_step/progress.csv") %>% add_column(name =  "1 step target", type = "Policy inference")
target_policy_3 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-target/independant_loss_3_step/progress.csv") %>% add_column(name =  "3 step target", type = "Policy inference")

policy_fixed_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/1/progress.csv") %>% add_column(name =  "1%", type = "Policy inference")
policy_fixed_5 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/5/progress.csv") %>% add_column(name =  "5%", type = "Policy inference")
policy_fixed_20 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/20/progress.csv") %>% add_column(name =  "20%", type = "Policy inference")
policy_fixed_60 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/60/progress.csv") %>% add_column(name =  "60%", type = "Policy inference")

apex = read_csv("experiments/runs/apex/cluster-apex-default-nstep-1-larger-epsilon/LegalActionApex_Hanabi_2c5d4c4c_2020-03-13_21-04-05lyhfb2m_/progress.csv",col_names = T, cols) %>% add_column(name =  "Ape-X DQN")

plot_timesteps = function (...){
  data = bind_rows(...) %>% mutate(name = factor(name, levels = levels)) %>%
    select(name, timesteps_total, episode_reward_mean) %>%
    group_by(name) %>%
    nest() %>%
    transmute(
    y = map(data, function (b){c(0,predict(loess(episode_reward_mean ~ timesteps_total,data = b, span=0.05)))}),
    x = map(data, function (b){c(0,b$timesteps_total)/1e6})
    ) %>%
    unnest(c(y, x))
  min_timesteps = data %>%
    group_by(name) %>%
    summarise(max = max(x)) %>%
    filter(max >= 0) %>% pull(max) %>% min()
  plot = ggplot() +
    theme_classic() +
    theme(aspect.ratio=1, legend.position = c(0.85,0.85), legend.title = element_blank(), legend.background =  element_rect(fill="transparent"))+
    scale_y_continuous(limits = c(0,25), sec.axis = dup_axis(labels = NULL, name = NULL)) +
    scale_x_continuous(limits = c(0, min_timesteps), sec.axis = dup_axis(labels = NULL, name = NULL)) +
    labs(x = "Timesteps (million)", y = "Game score")
  plot = plot + geom_line(data, mapping = aes(x = x, y = y, color = name)) +
    scale_color_manual(values = cbp2)
  plot
}

plot_timesteps(baseline)
plot_timesteps(baseline, apex)
plot_timesteps(baseline, stack_round_1, stack_round_3, stack_round_6, stack_round_1_deeper)
plot_timesteps(baseline, hand_adaptive, hand_1, hand_5, hand_20, hand_60)
plot_timesteps(baseline, policy_1, policy_3, target_policy_3, target_policy_1)
plot_timesteps(baseline, policy_1 %>% mutate(name = "Adaptive"), policy_fixed_1, policy_fixed_5, policy_fixed_20, policy_fixed_60)

ggsave("experiments/plots/training/baseline_training.pdf", plot = plot_timesteps(baseline), width = 6, height = 6)
ggsave("experiments/plots/training/apex_training.pdf", plot = plot_timesteps(baseline, apex), width = 6, height = 6)
ggsave("experiments/plots/training/round_stacking_training.pdf", plot = plot_timesteps(baseline, stack_round_1, stack_round_3, stack_round_6, stack_round_1_deeper), width = 6, height = 6)
ggsave("experiments/plots/training/hand_inference_training.pdf", plot = plot_timesteps(baseline, hand_adaptive, hand_1, hand_5, hand_20, hand_60), width = 6, height = 6)
ggsave("experiments/plots/training/policy_inference_training.pdf", plot = plot_timesteps(baseline, policy_1, policy_3, target_policy_3, target_policy_1), width = 6, height = 6)
ggsave("experiments/plots/training/policy_inference_fixed_training.pdf", plot = plot_timesteps(baseline, policy_1 %>% mutate(name = "Adaptive"), policy_fixed_1, policy_fixed_5, policy_fixed_20, policy_fixed_60), width = 6, height = 6)


plot_loss = function (...){
  data = bind_rows(...) %>% mutate(name = factor(name, levels = levels)) %>%
    select(name, timesteps_total, loss) %>%
    group_by(name) %>%
    nest() %>%
    transmute(
    y = map(data, function (b){predict(loess(loss ~ timesteps_total,data = b, span=0.05))}),
    x = map(data, function (b){b$timesteps_total/1e6})
    ) %>%
    unnest(c(y, x))
  min_timesteps = data %>%
    group_by(name) %>%
    summarise(max = max(x)) %>%
    filter(max >= 0) %>% pull(max) %>% min()
  plot = ggplot() +
    theme_classic() +
    theme(aspect.ratio=1, legend.position = c(0.85,0.85), legend.title = element_blank(), legend.background =  element_rect(fill="transparent"))+
    scale_y_continuous(limits = c(0,3), sec.axis = dup_axis(labels = NULL, name = NULL)) +
    scale_x_continuous(limits = c(0, min_timesteps), sec.axis = dup_axis(labels = NULL, name = NULL)) +
    labs(x = "Timesteps (million)", y = "Auxiliary loss")
  plot = plot + geom_line(data, mapping = aes(x = x, y = y, color = name)) +
    scale_color_manual(values = cbp2[-1])
  plot
}

plot_loss(bind_rows(policy_1, policy_3, target_policy_3, target_policy_1) %>% transmute(loss = `info/learner/default_policy/policy_inference_loss`, timesteps_total, name))
plot_loss(bind_rows(policy_1 %>% mutate(name = "Adaptive"), policy_fixed_1, policy_fixed_5, policy_fixed_20, policy_fixed_60) %>% transmute(loss = `info/learner/default_policy/policy_inference_loss`, timesteps_total, name))
plot_loss(bind_rows(hand_adaptive, hand_1, hand_5, hand_20, hand_60) %>% transmute(loss = `info/learner/default_policy/hand_inference_loss`, timesteps_total, name))

ggsave("experiments/plots/training/policy_inference_loss.pdf", plot = plot_loss(bind_rows(policy_1, policy_3, target_policy_3, target_policy_1) %>% transmute(loss = `info/learner/default_policy/policy_inference_loss`, timesteps_total, name)), width = 6, height = 6)
ggsave("experiments/plots/training/policy_inference_fixed_loss.pdf", plot = plot_loss(bind_rows(policy_1 %>% mutate(name = "Adaptive"), policy_fixed_1, policy_fixed_5, policy_fixed_20, policy_fixed_60) %>% transmute(loss = `info/learner/default_policy/policy_inference_loss`, timesteps_total, name)), width = 6, height = 6)
ggsave("experiments/plots/training/hand_inference_loss.pdf", plot = plot_loss(bind_rows(hand_adaptive, hand_1, hand_5, hand_20, hand_60) %>% transmute(loss = `info/learner/default_policy/hand_inference_loss`, timesteps_total, name)), width = 6, height = 6)

plot_time = function (...){
  data = bind_rows(...) %>% mutate(name = factor(name, levels = levels)) %>%
    select(name, time_total_s, episode_reward_mean) %>%
    group_by(name) %>%
    nest() %>%
    transmute(
    y = map(data, function (b){c(0,predict(loess(episode_reward_mean ~ time_total_s,data = b, span=0.05)))}),
    x = map(data, function (b){c(0,b$time_total_s)/ 3600 /24 })
    ) %>%
    unnest(c(y, x))
  plot = ggplot() +
    theme_classic() +
    theme(aspect.ratio=1, legend.position = c(0.85,0.85), legend.title = element_blank(), legend.background =  element_rect(fill="transparent"))+
    scale_y_continuous(limits = c(0,25), sec.axis = dup_axis(labels = NULL, name = NULL)) +
    scale_x_continuous(breaks = seq(0,7, 1), sec.axis = dup_axis(labels = NULL, name = NULL)) +
    labs(x = "Training time (days)", y = "Game score")
  plot = plot + geom_line(data, mapping = aes(x = x, y = y, color = name)) +
    scale_color_manual(values = cbp2)
  plot
}

plot_time(baseline, apex)

ggsave("experiments/plots/training/apex_time_training.pdf", plot = plot_time(baseline, apex), width = 6, height = 6)
