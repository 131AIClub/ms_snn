import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.functional as F


class VotingLayer(nn.Cell):
    def __init__(self, voting_size: int = 10):  # , step_mode='s'

        super().__init__()
        self.voting_size = voting_size

    @ms.jit
    def construct(self, x: ms.Tensor):
        x = x.expand_dims(0)
        return F.avg_pool1d(x, self.voting_size, self.voting_size).squeeze(0)


if __name__ == '__main__':
    import numpy as np
    import mindspore as ms
    import torch
    from mindspore import context
    from spikingjelly.activation_based import layer
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    # 加载数据并创建需要梯度的张量
    array = np.random.randn(1000, 1000)*1e10
    array_ms = ms.Tensor(array, dtype=ms.float32)
    array_torch = torch.tensor(array, dtype=torch.float32, requires_grad=True)

    # mindspore部分
    atan_ms = VotingLayer()

    def net(x):
        return atan_ms(x).mean()
    dout_ms = ms.grad(net, grad_position=0)(array_ms)

    # torch部分
    atan_torch = layer.VotingLayer()
    result = atan_torch(array_torch).mean()
    result.backward()
    dout_torch = array_torch.grad

    if not np.allclose(dout_ms.numpy(), dout_torch.numpy(), rtol=1e-5):
        print("bad implement for file "+__file__)
