# Title     : rollout plots
# Objective : generate plots and analysis for the rollouts
# Created by: carl
# Created on: 27.03.20

setwd("/home/carl/PycharmProjects/HanabiExperiments")
library("tidyverse")

baseline = read_csv("experiments/runs/dqn/local-dqn-deepmind/LegalActionDQN_Hanabi_a47106c6_2020-01-26_21-15-48bkgpc4fw/checkpoint_9999/rollout_episode_rewards.txt",col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "baseline")

stack_round_1 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/1_turn/checkpoint_3000/rollout_episode_rewards.txt",col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1 round")
stack_round_3 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/3_turn/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "3 rounds")
stack_round_6 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/6_turn/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "6 rounds")
stack_round_1_deeper = read_csv("experiments/runs/dqn/local-dqn-turn-stacking-deeper/LegalActionDQN_Hanabi_0f119e68_2020-03-14_11-46-28hpd71rwd/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1 round deeper")

hand_adaptive = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference/independant_loss/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "adaptive")
hand_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/1/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1%")
hand_5 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/5/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "5%")
hand_20 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/20/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "20%")
hand_60 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/60/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "60%")

policy_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference/independant_loss_1_step/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1 step")
policy_3 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference/independant_loss_3_step/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "3 step")
policy_1_target = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-target/independant_loss_1_step/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1 step target")
policy_3_target = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-target/independant_loss_3_step/checkpoint_3050/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "3 step target")

policy_fixed_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/1/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1%")
policy_fixed_5 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/5/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "5%")
policy_fixed_20 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/20/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "20%")
policy_fixed_60 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/60/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "60%")

apex = read_csv("experiments/runs/apex/cluster-apex-default-nstep-1-larger-epsilon/LegalActionApex_Hanabi_2c5d4c4c_2020-03-13_21-04-05lyhfb2m_/checkpoint_2866/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "Ape-X DQN")


base_plot = ggplot( mapping = aes(x = game_score, y = (..count..)/sum(..count..))) + lims(x = c(0,25)) + labs(x = "Game score", y = "Proportion of games") +theme_linedraw() + theme(aspect.ratio=1, legend.position = c(0.15,0.85), legend.title = element_blank())

baseline_plot = base_plot +
  geom_histogram(data = baseline, binwidth = 0) +
  geom_step(data = baseline %>% group_by(game_score) %>% summarise(count = n()) %>% mutate(p = count/sum(count)), aes(x = game_score, y = p))
ggsave("baseline_rollout_histogram.pdf", plot = baseline_plot, width = 5, height = 5)

round_stacking_plot = base_plot +
  geom_line(data = bind_rows(stack_round_1, stack_round_3, stack_round_6, stack_round_1_deeper), mapping = aes(color = name))
ggsave("turn_stacking_rollout_histogram.pdf", plot = baseline_plot, width = 5, height = 5)

ggplot() +
  geom_step(data = baseline %>% group_by(game_score) %>% summarise(count = n()) %>% mutate(p = count/sum(count)) %>% complete(game_score = seq(0,25), fill = list(p = 0)), aes(x = game_score, y=p))
