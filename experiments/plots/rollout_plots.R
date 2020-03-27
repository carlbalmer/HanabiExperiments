# Title     : rollout plots
# Objective : generate plots and analysis for the rollouts
# Created by: carl
# Created on: 27.03.20

setwd("/home/carl/PycharmProjects/HanabiExperiments")

library("tidyverse")

baseline = read_csv("experiments/runs/dqn/local-dqn-deepmind/LegalActionDQN_Hanabi_a47106c6_2020-01-26_21-15-48bkgpc4fw/checkpoint_9999/rollout_episode_rewards.txt",col_names = c("game_score")) %>% add_column(name =  "baseline") %>% mutate(game_score = as.integer(game_score))

base_plot = ggplot( mapping = aes(x = game_score, y = (..count..)/sum(..count..))) + lims(x = c(0,25)) + labs(x = "Game score", y = "Proportion of games") +theme_linedraw() + theme(aspect.ratio=1, legend.position = c(0.15,0.85), legend.title = element_blank())

baseline_plot = base_plot +
  geom_histogram(data = baseline, binwidth = 1)
ggsave("baseline_rollout_histogram.pdf", plot = baseline_plot, width = 5, height = 5)

