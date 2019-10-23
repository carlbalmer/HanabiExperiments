import numpy
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector


class MultiAgentCollectorMixin:

    def collect_batch(self, agent_inputs, traj_infos, itr):
        results = super(MultiAgentCollectorMixin, self).collect_batch(agent_inputs, traj_infos, itr)
        return results


def rolling_sum(a, n=4) :
    ret = numpy.cumsum(a, axis=1, dtype=float)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:]

class MultiAgentCpuResetCollector(MultiAgentCollectorMixin, CpuResetCollector):
    pass