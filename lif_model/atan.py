import math
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import Custom, CustomRegOp, DataType


class ATan(nn.Cell):
    def __init__(self, alpha=2.0):
        super(ATan, self).__init__()
        self.alpha = alpha

        aot_bprop_info = CustomRegOp() \
            .input(0, "x") \
            .input(1, "dout") \
            .output(0, "dx") \
            .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
            .attr("alpha", "required", "float", value=alpha) \
            .target("GPU") \
            .get_op_info()
        op_info = CustomRegOp() \
            .input(0, "x") \
            .output(0, "out") \
            .dtype_format(DataType.F32_Default, DataType.F32_Default) \
            .target("GPU") \
            .get_op_info()

        aot_bprop = Custom(func="/home/lwb/ms_dataset/lif_model/ATan.so:AtanBackward", out_dtype=lambda x, _: x,
                           out_shape=lambda x, _: x, func_type="aot", reg_info=aot_bprop_info)

        def bprop(x, out, dout):
            res = aot_bprop(x, dout)
            # if ms.numpy.isnan(res).any() or ms.numpy.isinf(res).any():
            #     print("-----------------------------x--------------------------------")
            #     print(x)
            #     print("-----------------------------out--------------------------------")
            #     print(out)
            #     print(
            #         "-----------------------------dout--------------------------------")
            #     print(dout)
            return (res,)

        self.op = Custom(func="/home/lwb/ms_dataset/lif_model/ATan.so:AtanForward", out_dtype=lambda x: x,
                         out_shape=lambda x: x, bprop=bprop, func_type="aot", reg_info=op_info)

        # def atan_forward(x: ms.Tensor):
        #     return (x >= 0).astype(x.dtype)

        # def atan_backword(x: ms.Tensor, out: ms.Tensor, dout: ms.Tensor):
        #     return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow(2)) * dout
        # self.op = Custom(
        #     func=atan_forward, out_dtype=lambda x: x, out_shape=lambda x: x, bprop=atan_backword, func_type="pyfunc")

    def construct(self, x):
        return self.op(x)


if __name__ == '__main__':
    # import torch
    # from spikingjelly.activation_based import surrogate
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x = ms.Tensor([-1.0, 0.0, 1.0], ms.float32)
    atan = ATan()
    print("mindspore result:")
    print(atan(x))
    print(ms.value_and_grad(atan)(x)[1])

    # x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    # atan = surrogate.ATan()
    # print("torch result:")
    # y = atan(x)
    # print(y)
    # y.sum().backward(retain_graph=True)
    # print(x.grad)
