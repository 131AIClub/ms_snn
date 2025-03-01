from typing import Callable
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import copy

from ..surrogate.atan import ATan


class BaseNode(nn.Cell):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = ATan(), store_v_seq=False, detach_reset=True):
        super(BaseNode, self).__init__()

        # 初始化记忆
        self._memories = {}
        self._memories_rv = {}

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.store_v_seq = store_v_seq

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    def register_memory(self, name, value):
        assert not hasattr(
            self, name), f'{name} has been set as a member variable!'
        self._memories[name] = value
        self.set_reset_value(name, value)

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def reset(self):
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def construct(self, x: ms.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def v_float_to_tensor(self, x: ms.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = ops.full_like(x, v_init)

    def neuronal_charge(self, x):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v-self.v_threshold)

    def neuronal_reset(self, spike: ms.Tensor):
        if self.detach_reset:
            spike_d = ops.stop_gradient(spike)
        else:
            spike_d = spike

        if self.v_reset is None:
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)
        else:
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    @staticmethod
    @ms.jit
    def jit_hard_reset(v: ms.Tensor, spike: ms.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @ms.jit
    def jit_soft_reset(v: ms.Tensor, spike: ms.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v
