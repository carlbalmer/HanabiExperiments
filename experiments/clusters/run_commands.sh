#!/usr/bin/env bash

ray exec --start --stop --tmux {cluster_config} -n {cluster_name} 'python run.py -f {experiment_config} --ray-address auto'
ray exec {cluster_config} -n {cluster_name} 'tensorboard --logdir ~/ray_results/ --port 6007' --port-forward 6007 --port-forward 8265
ray attach {cluster_config} -n {cluster_name} --tmux