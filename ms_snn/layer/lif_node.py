from typing import Callable
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from ..surrogate.atan import ATan
from .base_node import BaseNode


class LIFNode(BaseNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = ATan(), store_v_seq=False, decay_input=True, detach_reset=True):
        super(LIFNode, self).__init__(v_threshold, v_reset, surrogate_function,
                                      detach_reset, store_v_seq)

        self.v = 0
        self.tau = tau
        self.decay_input = decay_input

    def v_float_to_tensor(self, x: ms.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = ops.full_like(x, v_init)

    def neuronal_charge(self, x):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_decay_input_reset0(
                    x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_decay_input(
                    x, self.v, self.v_reset, self.tau)

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_no_decay_input_reset0(
                    x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_no_decay_input(
                    x, self.v, self.v_reset, self.tau)

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
    def neuronal_charge_no_decay_input_reset0(x: ms.Tensor, v: ms.Tensor, tau: float):
        v = v * (1. - 1. / tau) + x
        return v

    @staticmethod
    @ms.jit
    def neuronal_charge_no_decay_input(x: ms.Tensor, v: ms.Tensor, v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        return v

    @staticmethod
    @ms.jit
    def neuronal_charge_decay_input_reset0(x: ms.Tensor, v: ms.Tensor, tau: float):
        v = v + (x - v) / tau
        return v

    @staticmethod
    @ms.jit
    def neuronal_charge_decay_input(x: ms.Tensor, v: ms.Tensor, v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    @ms.jit
    def jit_eval_single_step_forward_hard_reset_decay_input(x: ms.Tensor, v: ms.Tensor, v_threshold: float,
                                                            v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        spike = (v >= v_threshold)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @ms.jit
    def jit_eval_single_step_forward_hard_reset_no_decay_input(x: ms.Tensor, v: ms.Tensor, v_threshold: float,
                                                               v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        spike = (v >= v_threshold)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @ms.jit
    def jit_eval_single_step_forward_soft_reset_decay_input(x: ms.Tensor, v: ms.Tensor, v_threshold: float,
                                                            tau: float):
        v = v + (x - v) / tau
        spike = (v >= v_threshold)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @ms.jit
    def jit_eval_single_step_forward_soft_reset_no_decay_input(x: ms.Tensor, v: ms.Tensor, v_threshold: float,
                                                               tau: float):
        v = v * (1. - 1. / tau) + x
        spike = (v >= v_threshold)
        v = v - spike * v_threshold
        return spike, v

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


if __name__ == '__main__':
    import numpy as np
    import mindspore as ms
    import torch
    from mindspore import context
    from spikingjelly.activation_based import neuron, surrogate
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    # 加载数据并创建需要梯度的张量
    array = np.random.randn(10, 10)*1000
    array_ms = ms.Tensor(array, dtype=ms.float32)
    array_torch = torch.tensor(array, dtype=torch.float32, requires_grad=True)

    # mindspore部分
    lif_node_ms = LIFNode()
    lif_node_ms.set_train(True)
    result_ms = lif_node_ms(array_ms)
    lif_node_ms.set_train(False)
    result_ms_eval = lif_node_ms(array_ms)

    def net(x):
        return lif_node_ms(x).mean()
    dout_ms = ms.grad(net, grad_position=0)(array_ms)

    # torch部分
    lif_node_torch = neuron.LIFNode(tau=2., v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(
    ), store_v_seq=False, decay_input=True, detach_reset=True)
    lif_node_torch.train()
    result = lif_node_torch(array_torch)
    result_torch = result.detach().numpy()
    lif_node_torch.eval()
    result_torch_eval = lif_node_torch(array_torch)
    result.mean().backward()
    dout_torch = array_torch.grad
    np.set_printoptions(threshold=np.inf)
    if not np.allclose(result_ms.numpy(), result_torch, rtol=1e-5):
        print("bad forward implement for file "+__file__)
    if not np.allclose(result_ms_eval.numpy(), result_torch_eval, rtol=1e-5):
        print("bad eval forward implement for file "+__file__)
