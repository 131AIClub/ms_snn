from typing import Callable
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from ..sorrogate.atan import ATan


class LIFNode(nn.Cell):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = ATan(), store_v_seq: bool = False):
        super(LIFNode, self).__init__()

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.surrogate_function = surrogate_function

        self.store_v_seq = store_v_seq

        # used in lava_exchange
        self.lava_s_cale = 1 << 6

        self.tau = tau
        self.v = None

    def construct(self, x: ms.Tensor):
        self.v_float_to_tensor(x)
        if self.training:
            return self.construct_train(x)
        else:
            spike, self.v = self.construct_test(
                x, self.v, self.v_threshold, self.v_reset, self.tau)
            return spike

    def construct_train(self, x: ms.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def construct_test(self, x: ms.Tensor, v: ms.Tensor, v_threshold: float, v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        spike = (v >= v_threshold).astype(v.dtype)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    def v_float_to_tensor(self, x: ms.Tensor):
        if self.v == None:
            v_init = 0
            self.v = ops.full_like(x, v_init)

    def neuronal_charge(self, x):
        self.v = self.v + (x - self.v) / self.tau

    def neuronal_fire(self):
        return self.surrogate_function(self.v-self.v_threshold)

    def neuronal_reset(self, spike):
        spike_d = ops.stop_gradient(spike)
        self.v = (1.-spike_d)*self.v+spike_d*self.v_reset


if __name__ == '__main__':
    # import torch
    # from spikingjelly.activation_based import neuron, surrogate
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x = ms.Tensor([-11.0, -5.0, -8.0], ms.float32)
    lif_node = LIFNode()
    lif_node.training = True
    print("mindspore train result:")
    print(lif_node(x))
    print(ms.value_and_grad(lif_node)(x)[1])

    # x = torch.tensor([-11.0, -5.0, -8.0], requires_grad=True)
    # lif_node = neuron.LIFNode(
    #     surrogate_function=surrogate.ATan(), detach_reset=True)
    # print("torch train result:")
    # y = lif_node(x)
    # y_grad = torch.autograd.grad(
    #     y, x, grad_outputs=torch.ones_like(y), retain_graph=True)
    # print(y)
    # print(y_grad)
