import ray
import yaml
import argparse
import rayExperiments
from ray import tune


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ray-address",
        default=None,
        type=str,
        help="Connect to an existing Ray cluster at this address instead "
        "of starting a new one.")
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    parser.add_argument(
        "--debug",
        default=False,
        type=bool,
        help="If true set ray to local node and use eager execution"
    )
    return parser


def run(args, parser):
    with open(args.config_file) as f:
        experiments = yaml.safe_load(f)
    ray_config = {}
    if args.ray_address:
        ray_config.update({"address": args.ray_address})
    if args.debug:
        ray_config.update({"local_mode": True})
        for exp in experiments.values():
            exp["config"].update({"eager": True})
    ray.init(**ray_config)
    tune.run_experiments(experiments)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
