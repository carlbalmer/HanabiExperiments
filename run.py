import ray
import yaml
import argparse
import HanabiExperiments
from ray import tune
import ray.tune.schedulers
import ray.tune.ray_trial_executor

# double the number of start attempts so autoscaler has time to restart workers
ray.tune.ray_trial_executor.TRIAL_START_ATTEMPTS = 10


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


def run(args, parser):
    ray_config = {}
    tune_args = {}

    with open(args.config_file) as f:
        experiments = yaml.safe_load(f)

    if "scheduler" in experiments:
        scheduler_config = experiments.pop("scheduler")
        scheduler = getattr(ray.tune.schedulers, scheduler_config["scheduler_cls"])(**scheduler_config["scheduler_args"])
    else:
        scheduler = None

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
    tune.run_experiments(experiments, scheduler=scheduler, **tune_args)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
