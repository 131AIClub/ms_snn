from mindspore.ops import Custom, CustomRegOp, DataType
from .base import SurrogateBase


class LeakyKReLU(SurrogateBase):
    def __init__(self, leak=0., k=1.):
        super(LeakyKReLU, self).__init__()

        aot_bprop_info = CustomRegOp() \
            .input(0, "x") \
            .input(1, "dout") \
            .output(0, "dx") \
            .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
            .attr("leak", "required", "float", value=leak) \
            .attr("k", "required", "float", value=k) \
            .target("GPU") \
            .get_op_info()

        aot_bprop = SurrogateBase.get_aot_op(
            "LeakyKReLUBackward", aot_bprop_info, lambda x, _: x, lambda x, _: x)

        def bprop(x, out, dout):
            res = aot_bprop(x, dout)
            return (res,)

        self.op = self.get_forward_op(bprop=bprop)
