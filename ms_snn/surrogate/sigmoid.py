from mindspore.ops import CustomRegOp, DataType
from .base import SurrogateBase


class Sigmoid(SurrogateBase):
    def __init__(self, alpha=4.0):
        super(Sigmoid, self).__init__()

        aot_bprop_info = CustomRegOp() \
            .input(0, "x") \
            .input(1, "dout") \
            .output(0, "dx") \
            .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
            .attr("alpha", "required", "float", value=alpha) \
            .target("GPU") \
            .get_op_info()

        aot_bprop = SurrogateBase.get_aot_op(
            "SigmoidBackward", aot_bprop_info, lambda x, _: x, lambda x, _: x)

        def bprop(x, out, dout):
            res = aot_bprop(x, dout)
            return (res,)

        self.op = self.get_forward_op(bprop=bprop)


if __name__ == '__main__':
    import numpy as np
    import mindspore as ms
    import torch
    from mindspore import context
    from spikingjelly.activation_based import surrogate
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    # 加载数据并创建需要梯度的张量
    array = np.random.randn(1000, 1000)*1000
    array_ms = ms.Tensor(array, dtype=ms.float32)
    array_torch = torch.tensor(array, dtype=torch.float32, requires_grad=True)

    # mindspore部分
    sigmoid_ms = Sigmoid()

    def net(x):
        return sigmoid_ms(x).mean()
    dout_ms = ms.grad(net, grad_position=0)(array_ms)

    # torch部分
    sigmoid_torch = surrogate.Sigmoid()
    result = sigmoid_torch(array_torch).mean()
    result.backward()
    dout_torch = array_torch.grad

    if not np.allclose(dout_ms.numpy(), dout_torch.numpy(), rtol=1e-5):
        print("bad implement for file "+__file__)
        print(f"mindspore: {dout_ms}")
        print(f"torch: {dout_torch}")
