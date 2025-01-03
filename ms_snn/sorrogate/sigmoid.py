from mindspore.ops import Custom, CustomRegOp, DataType
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

        aot_bprop = SurrogateBase.get_aot_op("Sigmoid", aot_bprop_info)

        def bprop(x, out, dout):
            res = aot_bprop(x, dout)
            return (res,)

        self.op = self.get_forward_op(bprop=bprop)
