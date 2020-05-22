# Title     : rollout plots
# Objective : generate plots and analysis for the rollouts
# Created by: carl
# Created on: 27.03.20

setwd("/home/carl/PycharmProjects/HanabiExperiments")
library("tidyverse")
levels <- c("Baseline", "Ape-X DQN", "1%", "5%", "20%", "60%", "Adaptive", "1 round", "3 rounds", "6 rounds", "1 round deeper", "1 step", "3 step", "1 step target", "3 step target")


baseline = read_csv("experiments/runs/dqn/local-dqn-deepmind/LegalActionDQN_Hanabi_a47106c6_2020-01-26_21-15-48bkgpc4fw/checkpoint_9999/rollout_episode_rewards.txt",col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "Baseline")

stack_round_1 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/1_turn/checkpoint_3000/rollout_episode_rewards.txt",col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1 round", type = "Round Stacking")
stack_round_3 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/3_turn/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "3 rounds", type = "Round Stacking")
stack_round_6 = read_csv("experiments/runs/dqn/local-dqn-turn-stacking/6_turn/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "6 rounds", type = "Round Stacking")
stack_round_1_deeper = read_csv("experiments/runs/dqn/local-dqn-turn-stacking-deeper/LegalActionDQN_Hanabi_0f119e68_2020-03-14_11-46-28hpd71rwd/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1 round deeper", type = "Round Stacking")

hand_adaptive = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference/independant_loss/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "Adaptive", type = "Hand inference")
hand_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/1/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1%", type = "Hand inference")
hand_5 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/5/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "5%", type = "Hand inference")
hand_20 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/20/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "20%", type = "Hand inference")
hand_60 = read_csv("experiments/runs/dqn/local-dqn-auxtask-hand-inference-fixed-ratio/60/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "60%", type = "Hand inference")

policy_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference/independant_loss_1_step/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1 step", type = "Policy inference")
policy_3 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference/independant_loss_3_step/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "3 step", type = "Policy inference")
target_policy_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-target/independant_loss_1_step/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1 step target", type = "Policy inference")
target_policy_3 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-target/independant_loss_3_step/checkpoint_3050/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "3 step target", type = "Policy inference")

policy_fixed_1 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/1/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "1%", type = "Policy inference")
policy_fixed_5 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/5/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "5%", type = "Policy inference")
policy_fixed_20 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/20/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "20%", type = "Policy inference")
policy_fixed_60 = read_csv("experiments/runs/dqn/local-dqn-auxtask-policy-inference-fixed-ratio/60/checkpoint_3000/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "60%", type = "Policy inference")

apex = read_csv("experiments/runs/apex/cluster-apex-default-nstep-1-larger-epsilon/LegalActionApex_Hanabi_2c5d4c4c_2020-03-13_21-04-05lyhfb2m_/checkpoint_2866/rollout_episode_rewards.txt", col_names = c("game_score")) %>% mutate(game_score = as.integer(game_score)) %>% add_column(name =  "Ape-X DQN")

plot_histogram = function (...) {
  data = bind_rows(...) %>% mutate(name = factor(name, levels = levels))
  statistics = data %>% group_by(name) %>% summarise(mean = mean(game_score), median = median(game_score), sd = sd(game_score))
  histogram = data %>%
    group_by(name, game_score) %>%
    summarise(n = n()) %>%
    mutate(p = n / sum(n)) %>%
    ungroup()
  plot = ggplot() +
    theme_classic() +
    theme(aspect.ratio=1) +
    scale_y_continuous(sec.axis = dup_axis(labels = NULL, name = NULL)) +
    scale_x_continuous(breaks = seq(0,25, 5), sec.axis = dup_axis(labels = NULL, name = NULL)) +
    labs(x = "Game score (points)", y = "Proportion of games")
  plot = plot +
    geom_col(data = histogram, mapping = aes(x = game_score, y = p), fill = "Black")
  if(n_distinct(data$name) > 1){
    plot = plot +
      facet_wrap(vars(name), ncol = 2)
  }
  y_max = ggplot_build(plot)$layout$panel_scales_y[[1]]$range$range[2]
  plot = plot +
    geom_text(data = statistics,
              mapping = aes(label = sprintf("Mean score = %.2f\nMedian score = %.2f\ns.d. = %.2f", mean, median,sd),x = 25, y = y_max),
              hjust = 1,
              vjust = 1,
              position = position_nudge(x = 2, y = 0.04)
    )
  plot
}

plot_histogram(baseline)
plot_histogram(apex)
plot_histogram(stack_round_1, stack_round_3, stack_round_6, stack_round_1_deeper)
plot_histogram(hand_adaptive, hand_1, hand_5, hand_20, hand_60)
plot_histogram(policy_1, policy_3, target_policy_1, target_policy_3)
plot_histogram(policy_1 %>% mutate(name = "Adaptive"), policy_fixed_5,policy_fixed_1, policy_fixed_20, policy_fixed_60)

ggsave("experiments/plots/rollouts/baseline_histogram.pdf", plot = plot_histogram(baseline), width = 6, height = 6)
ggsave("experiments/plots/rollouts/apex_histogram.pdf", plot = plot_histogram(apex), width = 6, height = 6)
ggsave("experiments/plots/rollouts/round_stack_histogram.pdf", plot = plot_histogram(stack_round_1, stack_round_3, stack_round_6, stack_round_1_deeper), width = 6, height = 6)
ggsave("experiments/plots/rollouts/hand_inference_histogram.pdf", plot = plot_histogram(hand_adaptive, hand_1, hand_5, hand_20, hand_60), width = 6, height = 9)
ggsave("experiments/plots/rollouts/policy_inference_histogram.pdf", plot = plot_histogram(policy_1, policy_3, target_policy_1, target_policy_3), width = 6, height = 6)
ggsave("experiments/plots/rollouts/policy_inference_fixed_histogram.pdf", plot = plot_histogram(policy_1 %>% mutate(name = "Adaptive"), policy_fixed_5, policy_fixed_1, policy_fixed_20, policy_fixed_60), width = 6, height = 9)

summary = bind_rows(baseline, stack_round_1, stack_round_3, stack_round_6, stack_round_1_deeper, hand_adaptive, hand_1, hand_5, hand_20, hand_60, policy_1, policy_3, target_policy_1, target_policy_3, policy_fixed_1, policy_fixed_5, policy_fixed_20, policy_fixed_60, apex) %>%
  unite("variant", type, name, sep = " " ) %>%
  group_by(variant) %>%
  summarise(
  mean = mean(game_score),
  median = median(game_score),
  sd = sd(game_score),
  ) %>% left_join(
bind_rows(baseline, stack_round_1, stack_round_3, stack_round_6, stack_round_1_deeper, hand_adaptive, hand_1, hand_5, hand_20, hand_60, policy_1, policy_3, target_policy_1, target_policy_3, policy_fixed_1, policy_fixed_5, policy_fixed_20, policy_fixed_60, apex) %>%
  unite("variant", type, name, sep = " " ) %>%
  group_by(variant, game_score) %>%
  summarise(n = n()) %>%
  mutate(p = n / sum(n)) %>%
  filter(game_score == 0) %>%
  select(variant, p_0 = p),
by = "variant"
)

summary %>% write_csv("experiments/plots/rollouts/rollout_summary.csv")
