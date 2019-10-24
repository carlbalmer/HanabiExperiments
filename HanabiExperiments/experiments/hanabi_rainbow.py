from os.path import abspath, join, dirname

import rlpyt.utils.logging.context as logContext
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from rlpyt.runners.minibatch_rl import MinibatchRlEval

from HanabiExperiments.agents.hanabi import HanabiCatDqnAgent
from HanabiExperiments.envs.hanabi import WrappedHanabiEnv
from HanabiExperiments.samplers.collector import MultiAgentCpuResetCollector
from HanabiExperiments.samplers.sampler import MultiAgentSerialSampler

logContext.LOG_DIR = abspath(join(dirname(__file__), '../../data'))


def build_and_train(run_ID=0, cuda_idx=None):
    sampler = MultiAgentSerialSampler(
        EnvCls=WrappedHanabiEnv,
        env_kwargs=dict(),
        eval_env_kwargs=dict(),
        CollectorCls=MultiAgentCpuResetCollector,
        batch_T=10,  # Four time-steps per sampler iteration.
        batch_B=4,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = CategoricalDQN(min_steps_learn=1e3,
               double_dqn=True,
               prioritized_replay=True,
               n_step_return=3)  # Run with defaults.
    agent = HanabiCatDqnAgent(
        model_kwargs=dict(hidden_sizes=[512, 512]),

    )
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=100e6,
        log_interval_steps=1e4,
        affinity=dict(cuda_idx=cuda_idx),
    )
    name = "hanabi_rainbow"
    log_dir = name
    with logContext.logger_context(log_dir, run_ID, name, snapshot_mode="last"):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
