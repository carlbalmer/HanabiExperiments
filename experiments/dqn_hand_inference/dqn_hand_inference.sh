#!/usr/bin/env bash

ray exec --start --stop --tmux experiments/dqn_hand_inference/dqn_hand_inference_cluster.yaml 'python run.py -f experiments/dqn_hand_inference/dqn_hand_inference.yaml --ray-address auto'
ray exec experiments/dqn_hand_inference/dqn_hand_inference_cluster.yaml 'tensorboard --logdir ~/ray_results/ --port 6006' --port-forward 6006 --port-forward 8081
# ray attach experiments/dqn_hand_inference/dqn_hand_inference_cluster.yaml --tmux