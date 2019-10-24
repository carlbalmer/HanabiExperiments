import multiprocessing as mp

from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.worker import sampling_process
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging import logger

from HanabiExperiments.samplers.collector import MultiAgentCpuResetCollector


class MultiAgentSerialSampler(SerialSampler):

    """Uses same functionality as ParallelSampler but does not fork worker
    processes; can be easier for debugging (e.g. breakpoint() in master).  Use
    with collectors which sample actions themselves (e.g. under cpu
    category)."""

    def __init__(self, EnvCls, env_kwargs, batch_T, batch_B, *args, CollectorCls=MultiAgentCpuResetCollector, **kwargs):
        self._n_actors = EnvCls(**env_kwargs).n_actors
        super(MultiAgentSerialSampler, self).__init__(EnvCls, env_kwargs, batch_T, batch_B * self._n_actors, *args, CollectorCls=CollectorCls, **kwargs)

    def initialize(
            self,
            agent,
            affinity=None,
            seed=None,
            bootstrap_value=False,
            traj_info_kwargs=None,
            rank=0,
            world_size=1,
            ):
        B = int(self.batch_spec.B / self._n_actors)  # only change from SerialSampler
        envs = [self.EnvCls(**self.env_kwargs) for _ in range(B)]
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        agent.initialize(envs[0].spaces, share_memory=False,
            global_B=global_B, env_ranks=env_ranks)
        samples_pyt, samples_np, examples = build_samples_buffer(agent, envs[0],
            self.batch_spec, bootstrap_value, agent_shared=False,
            env_shared=False, subprocess=False)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
        collector = self.CollectorCls(
            rank=0,
            envs=envs,
            samples_np=samples_np,
            batch_T=self.batch_spec.T,
            TrajInfoCls=self.TrajInfoCls,
            agent=agent,
            global_B=global_B,
            env_ranks=env_ranks,  # Might get applied redundantly to agent.
        )
        if self.eval_n_envs > 0:  # May do evaluation.
            eval_envs = [self.EnvCls(**self.eval_env_kwargs)
                for _ in range(self.eval_n_envs)]
            eval_CollectorCls = self.eval_CollectorCls or SerialEvalCollector
            self.eval_collector = eval_CollectorCls(
                envs=eval_envs,
                agent=agent,
                TrajInfoCls=self.TrajInfoCls,
                max_T=self.eval_max_steps // self.eval_n_envs,
                max_trajectories=self.eval_max_trajectories,
            )

        agent_inputs, traj_infos = collector.start_envs(
            self.max_decorrelation_steps)
        collector.start_agent()

        self.agent = agent
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.collector = collector
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        logger.log("Serial Sampler initialized.")
        return examples


class MultiAgentParallelSamplerMixin:

    def __init__(self, EnvCls, env_kwargs, batch_T, batch_B, *args, **kwargs):
        self._n_actors = EnvCls(**env_kwargs).n_actors
        super(MultiAgentParallelSamplerMixin, self).__init__(EnvCls, env_kwargs, batch_T, batch_B * self._n_actors, *args, **kwargs)

    def initialize(
            self,
            agent,
            affinity,
            seed,
            bootstrap_value=False,
            traj_info_kwargs=None,
            world_size=1,
            rank=0,
            worker_process=None,
            ):
        B = int(self.batch_spec.B / self._n_actors)  # only change from ParallelBase
        n_envs_list = self._get_n_envs_list(affinity=affinity, B=B)
        self.n_worker = n_worker = len(n_envs_list)
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        self.world_size = world_size
        self.rank = rank

        if self.eval_n_envs > 0:
            self.eval_n_envs_per = max(1, self.eval_n_envs // n_worker)
            self.eval_n_envs = eval_n_envs = self.eval_n_envs_per * n_worker
            logger.log(f"Total parallel evaluation envs: {eval_n_envs}.")
            self.eval_max_T = eval_max_T = int(self.eval_max_steps // eval_n_envs)

        env = self.EnvCls(**self.env_kwargs)
        self._agent_init(agent, env, global_B=global_B,
            env_ranks=env_ranks)
        examples = self._build_buffers(env, bootstrap_value)
        env.close()
        del env

        self._build_parallel_ctrl(n_worker)

        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing every init.

        common_kwargs = self._assemble_common_kwargs(affinity, global_B)
        workers_kwargs = self._assemble_workers_kwargs(affinity, seed, n_envs_list)

        target = sampling_process if worker_process is None else worker_process
        self.workers = [mp.Process(target=target,
            kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs]
        for w in self.workers:
            w.start()

        self.ctrl.barrier_out.wait()  # Wait for workers ready (e.g. decorrelate).
        return examples  # e.g. In case useful to build replay buffer.

    def _assemble_workers_kwargs(self, affinity, seed, n_envs_list):
        workers_kwargs = list()
        i_env = 0
        g_env = sum(n_envs_list) * self.rank
        for rank in range(len(n_envs_list)):
            n_envs = n_envs_list[rank]
            slice_B = slice(i_env, i_env + n_envs*self._n_actors)
            env_ranks = list(range(g_env, g_env + n_envs))
            worker_kwargs = dict(
                rank=rank,
                env_ranks=env_ranks,
                seed=seed + rank,
                cpus=(affinity["workers_cpus"][rank]
                    if affinity.get("set_affinity", True) else None),
                n_envs=n_envs,
                samples_np=self.samples_np[:, slice_B],
                sync=self.sync,  # Only for eval, on CPU.
            )
            i_env += n_envs
            g_env += n_envs
            workers_kwargs.append(worker_kwargs)
        return workers_kwargs


class MultiAgentCpuSampler(MultiAgentParallelSamplerMixin, CpuSampler):

    def __init__(self, *args, CollectorCls=MultiAgentCpuResetCollector, **kwargs):
        super(MultiAgentCpuSampler, self).__init__(*args, CollectorCls=CollectorCls, **kwargs)
