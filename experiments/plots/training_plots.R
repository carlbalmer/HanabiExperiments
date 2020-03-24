# Title     : Training Plots
# Objective : Plot training progress for thesis
# Created by: carl
# Created on: 22.03.20
setwd("/home/carl/PycharmProjects/HanabiExperiments")

library("tidyverse")
cols = cols(experiment_tag = col_character())
baseline = read_csv("experiments/runs/dqn/local-dqn-deepmind/LegalActionDQN_Hanabi_a47106c6_2020-01-26_21-15-48bkgpc4fw/progress.csv",col_names = T, cols) %>% add_column(name =  "baseline")

stack_round_1 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/1_turn/progress.csv") %>% add_column(name =  "1 round")
stack_round_3 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/3_turn/progress.csv") %>% add_column(name =  "3 rounds")
stack_round_6 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/6_turn/progress.csv") %>% add_column(name =  "6 rounds")
stack_round_1_deeper = read_csv("experiments/runs/dqn/local-dqn-turn-stacking-deeper/LegalActionDQN_Hanabi_0f119e68_2020-03-14_11-46-28hpd71rwd/progress.csv", col_names = T, cols) %>% add_column(name =  "1 round deeper")

hand_adaptive = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference/independant_loss/progress.csv") %>% add_column(name =  "adaptive")
hand_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/1/progress.csv") %>% add_column(name =  "1%")
hand_5 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/5/progress.csv") %>% add_column(name =  "5%")
hand_20 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/20/progress.csv") %>% add_column(name =  "20%")
hand_60 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/60/progress.csv") %>% add_column(name =  "60%")

policy_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference/independant_loss_1_step/progress.csv") %>% add_column(name =  "1 step")
policy_3 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference/independant_loss_3_step/progress.csv") %>% add_column(name =  "3 step")
target_policy_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-target/independant_loss_1_step/progress.csv") %>% add_column(name =  "1 step target")
target_policy_3 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-target/independant_loss_3_step/progress.csv") %>% add_column(name =  "3 step target")

policy_fixed_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/1/progress.csv") %>% add_column(name =  "1%")
policy_fixed_5 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/5/progress.csv") %>% add_column(name =  "5%")
policy_fixed_20 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/20/progress.csv") %>% add_column(name =  "20%")
policy_fixed_60 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/60/progress.csv") %>% add_column(name =  "60%")

apex = read_csv("experiments/runs/apex/cluster-apex-default-nstep-1-larger-epsilon/LegalActionApex_Hanabi_2c5d4c4c_2020-03-13_21-04-05lyhfb2m_/progress.csv",col_names = T, cols) %>% add_column(name =  "Ape-X DQN")

base_plot = ggplot() + lims(y = c(0,25)) + labs(x = "timesteps (million)", y = "game score") +theme_linedraw() + theme(aspect.ratio=1, legend.position = c(0.15,0.85), legend.title = element_blank())

baseline_plot = base_plot +
  geom_line(data = baseline, mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean), color = "lightblue") +
  stat_smooth(data = baseline, mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean), color = "black", method = "loess", span = 0.05, n = 1000, se = F)

ggsave("baseline.pdf", plot = baseline_plot, width = 5, height = 5)

hand_inference_plot = base_plot +
  stat_smooth(data = bind_rows(hand_adaptive, hand_1, hand_5, hand_20, hand_60), mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean, color = name), method = "loess", span = 0.05, n = 1000, se = F) +
  stat_smooth(data = baseline, mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean), color = "black", method = "loess", span = 0.05, n = 1000, se = F) +
  lims(x = c(0,30))
ggsave("hand_inference.pdf", plot = hand_inference_plot, width = 5, height = 5)

policy_inference_plot = base_plot +
  stat_smooth(data = bind_rows(policy_1, policy_3, target_policy_3, target_policy_1), mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean, color = name), method = "loess", span = 0.05, n = 1000, se = F) +
  stat_smooth(data = baseline, mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean), color = "black", method = "loess", span = 0.05, n = 1000, se = F) +
  lims(x = c(0,30))
ggsave("policy_inference.pdf", plot = policy_inference_plot, width = 5, height = 5)

policy_inference_fixed_plot = base_plot +
  stat_smooth(data = bind_rows(policy_1 %>% mutate(name = "adaptive"), policy_fixed_1, policy_fixed_5, policy_fixed_20, policy_fixed_60), mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean, color = name), method = "loess", span = 0.05, n = 1000, se = F) +
  stat_smooth(data = baseline, mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean), color = "black", method = "loess", span = 0.05, n = 1000, se = F) +
  lims(x = c(0,30))
ggsave("policy_inference_fixed.pdf", plot = policy_inference_fixed_plot, width = 5, height = 5)

policy_inference_loss_plot = base_plot +
  stat_smooth(data = bind_rows(policy_1, policy_3, target_policy_3, target_policy_1),mapping = aes(x = timesteps_total / 1e6, y = `info/learner/default_policy/policy_inference_loss`, color = name), method = "loess", span = 0.05, n = 1000, se = F) +
  lims(x = c(0,30), y = c(0,4)) + labs(y = "auxiliary loss (cross entropy)")
ggsave("policy_inference_loss.pdf", plot = policy_inference_loss_plot, width = 5, height = 5)

policy_inference_fixed_loss_plot = base_plot +
  stat_smooth(data = bind_rows(policy_1 %>% mutate(name = "adaptive"), policy_fixed_1, policy_fixed_5, policy_fixed_20, policy_fixed_60), mapping = aes(x = timesteps_total / 1e6, y = `info/learner/default_policy/policy_inference_loss`, color = name), method = "loess", span = 0.05, n = 1000, se = F) +
  lims(x = c(0,30), y = c(0,4)) + labs(y = "auxiliary loss (cross entropy)")
ggsave("policy_inference_fixed_loss.pdf", plot = policy_inference_fixed_loss_plot, width = 5, height = 5)

hand_inference_loss_plot = base_plot +
  stat_smooth(data = bind_rows(hand_adaptive, hand_1, hand_5, hand_20, hand_60),  mapping = aes(x = timesteps_total / 1e6, y = `info/learner/default_policy/hand_inference_loss`, color = name), method = "loess", span = 0.05, n = 1000, se = F) +
  lims(x = c(0,30), y = c(0,4)) + labs(y = "auxiliary loss (cross entropy)")
ggsave("hand_inference_loss.pdf", plot = hand_inference_loss_plot, width = 5, height = 5)

apex_plot = base_plot +
  stat_smooth(data = apex, mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean, color = name), method = "loess", span = 0.05, n = 1000) +
  stat_smooth(data = baseline, mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean), color = "black", method = "loess", span = 0.05, n = 1000)
ggsave("apex.pdf", plot = apex_plot, width = 5, height = 5)

apex_time_plot = base_plot +
  stat_smooth(data = apex, mapping = aes(x = time_total_s / 3600 /24 , y = episode_reward_mean, color = name), method = "loess", span = 0.05, n = 1000) +
  stat_smooth(data = baseline, mapping = aes(x = time_total_s / 3600 / 24, y = episode_reward_mean), color = "black", method = "loess", span = 0.05, n = 1000) +
  labs(x = "training time (days)")
ggsave("apex_time.pdf", plot = apex_time_plot, width = 5, height = 5)

round_stacking_plot = base_plot +
  stat_smooth(data = bind_rows(stack_round_1, stack_round_3, stack_round_6, stack_round_1_deeper), mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean, color = name), method = "loess", span = 0.05, n = 1000, se = F) +
  stat_smooth(data = baseline, mapping = aes(x = timesteps_total / 1e6, y = episode_reward_mean), color = "black", method = "loess", span = 0.05, n = 1000, se = F) +
  lims(x = c(0,30))
ggsave("round_stacking.pdf", plot = round_stacking_plot, width = 5, height = 5)
