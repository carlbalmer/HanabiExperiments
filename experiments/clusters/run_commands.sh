#!/usr/bin/env bash

ray exec --start --stop --tmux experiments/runs/cluster/apex/default/cluster.yaml 'python run.py -f experiments/runs/cluster/apex/default/config.yaml --ray-address auto'
ray exec experiments/runs/cluster/apex/default/cluster.yaml 'tensorboard --logdir ~/ray_results/ --port 6006' --port-forward 6006 --port-forward 8081
# ray attach experiments/runs/cluster/apex/default/cluster.yaml --tmux