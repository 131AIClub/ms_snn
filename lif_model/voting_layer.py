import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.functional as F


class VotingLayer(nn.Cell):
    def __init__(self, voting_size: int = 10):  # , step_mode='s'

        super().__init__()
        self.voting_size = voting_size

    def construct(self, x: ms.Tensor):
        x = x.expand_dims(0)
        return F.avg_pool1d(x, self.voting_size, self.voting_size).squeeze(0)


if __name__ == '__main__':
    # import torch
    # from spikingjelly.activation_based import layer
    x = ms.Tensor([[1.0, 2.0, 3.0, 3.0, 2.0, 3.0]])
    voting_layer = VotingLayer(voting_size=3)
    print("mindspore result:")
    grad_fn = ms.value_and_grad(voting_layer)
    y, grads = grad_fn(x)
    print(y)
    print(grads)

    x = torch.tensor([[1.0, 2.0, 3.0, 3.0, 2.0, 3.0]], requires_grad=True)
    voting_layer = layer.VotingLayer(3)
    print("torch result:")
    y = voting_layer(x)
    print(y)
    y.sum().backward(retain_graph=True)
    print(x.grad)
