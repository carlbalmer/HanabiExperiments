#!/usr/bin/env bash

ray exec --start --stop --tmux experiments/apex_population/hanabi_apex_population_cluster.yaml 'python run.py -f experiments/apex_population/hanabi_apex_population.yaml --ray-address auto'
ray exec experiments/apex_population/hanabi_apex_population_cluster.yaml 'tensorboard --logdir ~/ray_results/ --port 6006' --port-forward 6006 --port-forward 8081
# ray experiments/apex_population/hanabi_apex_population_cluster.yaml --tmux