#!/usr/bin/env bash

ray exec --start --stop --tmux experiments/dqn_deepmind_baseline/hanabi_dqn_deepmind_cluster.yaml 'python run.py -f experiments/dqn_deepmind_baseline/hanabi_dqn_deepmind.yaml --ray-address auto'
ray exec experiments/dqn_deepmind_baseline/hanabi_dqn_deepmind_cluster.yaml 'tensorboard --logdir ~/ray_results/ --port 6006' --port-forward 6006 --port-forward 8081
# ray attach experiments/dqn_deepmind_baseline/hanabi_dqn_deephmind_cluster.yaml --tmux