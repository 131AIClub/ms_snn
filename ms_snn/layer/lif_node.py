from typing import Callable
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from ..surrogate.atan import ATan


class LIFNode(nn.Cell):
    def __init__(self, tau: float = 2., v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = ATan(), store_v_seq=False, decay_input=True, detach_reset=True):
        super(LIFNode, self).__init__()

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.surrogate_function = surrogate_function

        self.store_v_seq = store_v_seq
        self.decay_input = decay_input
        self.detach_reset = detach_reset

        self.tau = tau
        self.v = 0

    def construct(self, x: ms.Tensor):
        self.v_float_to_tensor(x)
        if self.training:
            return self.construct_train(x)
        else:
            if self.v_reset is None:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(x, self.v,
                                                                                             self.v_threshold, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.tau)
            else:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_decay_input(x, self.v,
                                                                                             self.v_threshold,
                                                                                             self.v_reset, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.v_reset,
                                                                                                self.tau)
        return spike

    def construct_train(self, x: ms.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

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
            spike_d = ms.tensor(spike.numpy())
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
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @ms.jit
    def jit_eval_single_step_forward_hard_reset_no_decay_input(x: ms.Tensor, v: ms.Tensor, v_threshold: float,
                                                               v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @ms.jit
    def jit_eval_single_step_forward_soft_reset_decay_input(x: ms.Tensor, v: ms.Tensor, v_threshold: float,
                                                            tau: float):
        v = v + (x - v) / tau
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @ms.jit
    def jit_eval_single_step_forward_soft_reset_no_decay_input(x: ms.Tensor, v: ms.Tensor, v_threshold: float,
                                                               tau: float):
        v = v * (1. - 1. / tau) + x
        spike = (v >= v_threshold).to(x)
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

    def net(x):
        return lif_node_ms(x).mean()
    dout_ms = ms.grad(net, grad_position=0)(array_ms)

    # torch部分
    lif_node_torch = neuron.LIFNode(tau=2., v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(
    ), store_v_seq=False, decay_input=True, detach_reset=True)
    lif_node_torch.train()
    result = lif_node_torch(array_torch)
    result_torch = result.detach().numpy()
    result.mean().backward()
    dout_torch = array_torch.grad
    np.set_printoptions(threshold=np.inf)
    if not np.allclose(result_ms.numpy(), result_torch, rtol=1e-5):
        print("bad forward implement for file "+__file__)
        print(f"mindspore: {result_ms.numpy()}")
        print(f"torch: {result_torch}")
        print(f"difference: {(result_torch==result_ms.numpy())}")

    if not np.allclose(dout_ms.numpy(), dout_torch.numpy(), rtol=1e-5):
        print("bad backword implement for file "+__file__)
        print(f"mindspore: {result_ms.numpy()}")
        print(f"torch: {result_torch}")
        print(f"mindspore: {dout_ms.numpy()}")
        print(f"torch: {dout_torch.numpy()}")
