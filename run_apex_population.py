import ray
import yaml
import argparse

from ray.tune.schedulers import PopulationBasedTraining

import HanabiExperiments
from ray import tune


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ray-address",
        default=None,
        type=str,
        help="Connect to an existing Ray cluster at this address instead "
             "of starting a new one."
    )
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
             "overrides any trial-specific options set via flags above."
    )
    parser.add_argument(
        "--debug",
        default=False,
        type=bool,
        help="If true set ray to local node and use eager execution"
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="One of “LOCAL”, “REMOTE”, “PROMPT”, or bool. LOCAL/True restores the checkpoint from the "
             "local_checkpoint_dir. REMOTE restores the checkpoint from remote_checkpoint_dir. PROMPT provides CLI "
             "feedback. False forces a new experiment. If resume is set but checkpoint does not exist, ValueError "
             "will be thrown."
    )
    return parser


pbt = PopulationBasedTraining(
    time_attr="timesteps_total",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=300000,
    quantile_fraction=0.35,
    hyperparam_mutations={
        "n_step": [1, 3, 5],
        # distribution for resampling
        "lr": [2.5e-5, 5e-4, 1e-5, 5e-5, 1e-4, 1e-3],
        # allow perturbations within this set of categorical values
        "adam_epsilon": [3.125e-5, 1e-8, 1e-5, 6e-5, 1e-4, 1e-3, 1e-6],
        "sample_batch_size": [25,50,100],
        "train_batch_size": [256, 512, 1024]
    })


def run(args, parser):
    with open(args.config_file) as f:
        experiments = yaml.safe_load(f)
    ray_config = {}
    tune_args = {}
    if args.ray_address:
        ray_config.update({"address": args.ray_address})
        if args.ray_address == "auto":
            tune_args.update({"queue_trials": True})
    if args.debug:
        ray_config.update({"local_mode": True})
        for exp in experiments.values():
            exp["config"].update({"eager": True})
    if args.resume:
        tune_args.update({"resume": args.resume})
    ray.init(**ray_config)
    tune.run_experiments(experiments, scheduler=pbt, **tune_args)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
