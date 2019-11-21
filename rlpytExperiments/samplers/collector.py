import numpy
from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.parallel.gpu.collectors import GpuResetCollector
from rlpyt.utils.buffer import numpify_buffer, torchify_buffer, buffer_from_example
from rlpyt.utils.collections import namedarraytuple_like


class MultiAgentCollectorBase:

    def __init__(self, *args, **kwargs):
        super(MultiAgentCollectorBase, self).__init__(*args, **kwargs)
        self._n_actors = self.envs[0].n_actors
        self._n_envs = len(self.envs)
        self._buffer_env_view = buffer_from_example(self.samples_np[0][0],
                                                    ((self.batch_T + 1) * self._n_actors, self._n_envs))
        self._buffer_actor_view = reshape_buffer(self._buffer_env_view,
                                                 (self.batch_T + 1, self._n_envs * self._n_actors))
        self.completed_infos = list()

    def start_envs(self, max_decorrelation_steps=0):
        agent_inputs, traj_infos = super(MultiAgentCollectorBase, self).start_envs(max_decorrelation_steps)
        agent_inputs, traj_infos, _ = self.record_steps_into_buffer(agent_inputs, traj_infos, 0,
                                                                    start=0,
                                                                    stop=self._n_actors)
        return agent_inputs, traj_infos

    def collect_batch(self, agent_inputs, traj_infos, itr):

        agent_inputs, traj_infos, completed_infos = self.record_steps_into_buffer(agent_inputs, traj_infos, itr,
                                                                                  start=self._n_actors,
                                                                                  stop=(self.batch_T + 1) * self._n_actors)
        self.copy_buffer_to_out()
        self.reset_buffer()

        if "bootstrap_value" in self.samples_np.agent:
            # agent.value() should not advance rnn state.
            self.samples_np.agent.bootstrap_value[:] = self.agent.value(*torchify_buffer(agent_inputs))

        return agent_inputs, traj_infos, completed_infos

    def record_steps_into_buffer(self, agent_inputs, traj_infos, itr, start, stop):
        raise NotImplementedError

    def set_prev_action_reward(self, action, reward, t):
        action[:] = 0
        reward[:] = 0
        index = t - self._n_actors
        done_at_t = self._buffer_env_view.env.done[index]
        action[:][~done_at_t] = self._buffer_env_view.agent.action[index][~done_at_t]
        reward[:][~done_at_t] = self._buffer_env_view.env.reward[index][~done_at_t]

    def copy_buffer_to_out(self):
        self.samples_np.agent.action[:] = self._buffer_actor_view.agent.action[:-1]
        self.samples_np.env.done[:] = self._buffer_actor_view.env.done[:-1]
        self.samples_np.env.observation[:] = self._buffer_actor_view.env.observation[:-1]
        self.samples_np.env.reward[:] = self._buffer_actor_view.env.reward[:-1]

        self.samples_np.agent.prev_action[0] = self._buffer_actor_view.agent.prev_action[0]
        self.samples_np.env.prev_reward[0] = self._buffer_actor_view.env.prev_reward[0]

        if self.samples_np.agent.agent_info:
            self.samples_np.agent.agent_info[:] = self._buffer_actor_view.agent.agent_info[:-1]
        if self.samples_np.env.env_info:
            self.samples_np.env.env_info[:] = self._buffer_actor_view.env.env_info[:-1]

    def reset_buffer(self):
        self._buffer_actor_view[0] = self._buffer_actor_view[-1]
        self._buffer_actor_view.agent.prev_action[0] = self._buffer_actor_view.agent.action[-2]
        self._buffer_actor_view.env.prev_reward[0] = self._buffer_actor_view.env.reward[-2]
        self.completed_infos = list()


class MultiAgentCpuResetCollector(MultiAgentCollectorBase, CpuResetCollector):

    def record_steps_into_buffer(self, agent_inputs, traj_infos, itr, start, stop):
        observation, action, reward = agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)

        self.agent.sample_mode(itr)

        for t in range(start, stop):
            self._buffer_env_view.env.observation[t] = observation

            act_pyt[:], agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)

            self._buffer_env_view.agent.action[t] = act_pyt
            if agent_info:
                self._buffer_env_view.agent.agent_info[t] = agent_info

            for b, env in enumerate(self.envs):
                index_last_action = t - self._n_actors + 1
                if index_last_action < 0:
                    index_last_action = 0

                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d, agent_info[b],
                                   env_info)

                self._buffer_env_view.env.reward.transpose()[b][index_last_action:t + 1][
                    ~self._buffer_env_view.env.done.transpose()[b][index_last_action:t + 1]] += r
                if getattr(env_info, "traj_done", d):
                    self.completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    self.agent.reset_one(idx=b)
                    self._buffer_env_view.env.done.transpose()[b][index_last_action:t + 1] = d
                observation[b] = o
                if env_info:
                    self._buffer_env_view.env_info[t, b] = env_info

            self.set_prev_action_reward(action, reward, t + 1)

        return AgentInputs(observation, action, reward), traj_infos, self.completed_infos.copy()


class MultiAgentGpuResetCollector(MultiAgentCollectorBase, GpuResetCollector):

    def record_steps_into_buffer(self, agent_inputs, traj_infos, itr, start, stop):
        self.sync.obs_ready.release()
        for t in range(start, stop):
            self._buffer_env_view.env.observation[t] = self.step_buffer_np.observation
            self.sync.act_ready.acquire()

            self._buffer_env_view.agent.action[t] = self.step_buffer_np.action
            if self.step_buffer_np.agent_info:
                self._buffer_env_view.agent.agent_info[t] = self.step_buffer_np.agent_info

            for b, env in enumerate(self.envs):
                index_last_action = t - self._n_actors + 1
                if index_last_action < 0:
                    index_last_action = 0

                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(self.step_buffer_np.action[b])
                traj_infos[b].step(self.step_buffer_np.observation[b], self.step_buffer_np.action[b], r, d, self.step_buffer_np.agent_info[b],
                                   env_info)

                self._buffer_env_view.env.reward.transpose()[b][index_last_action:t + 1][
                    ~self._buffer_env_view.env.done.transpose()[b][index_last_action:t + 1]] += r
                if getattr(env_info, "traj_done", d):
                    self.completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    self._buffer_env_view.env.done.transpose()[b][index_last_action:t + 1] = d
                self.step_buffer_np.observation[b] = o
                self.step_buffer_np.done[b] = d
                if env_info:
                    self._buffer_env_view.env_info[t, b] = env_info
            self.set_prev_action_reward(self.step_buffer_np.action, self.step_buffer_np.reward, t + 1)
            self.sync.obs_ready.release()

        return None, traj_infos, self.completed_infos.copy()


def reshape_buffer(buffer, leading_dims):
    if buffer is None:
        return
    try:
        buffer_type = namedarraytuple_like(buffer)
    except TypeError:  # example was not a namedtuple or namedarraytuple
        return numpy.reshape(buffer, leading_dims + buffer.shape[len(leading_dims):])
    return buffer_type(*(reshape_buffer(v, leading_dims) for v in buffer))
