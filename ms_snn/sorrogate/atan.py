from mindspore.ops import CustomRegOp, DataType
from importlib.resources import path

from .base import SurrogateBase


class ATan(SurrogateBase):
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

        aot_bprop = SurrogateBase.get_aot_op(
            "AtanBackward", aot_bprop_info, lambda x, _: x, lambda x, _: x)

        def bprop(x, out, dout):
            res = aot_bprop(x, dout)
            return (res,)

        self.op = self.get_forward_op(bprop=bprop)
