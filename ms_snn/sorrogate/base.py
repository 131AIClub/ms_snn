import mindspore.nn as nn
from mindspore.ops import Custom, CustomRegOp, DataType
from importlib.resources import files, as_file


class SurrogateBase(nn.Cell):
    def __init__(self):
        super(SurrogateBase, self).__init__()

    @staticmethod
    def get_aot_op(name: str, op_info, bprop=None):
        pkg = files("ms_snn")
        with as_file(pkg/"cuda"/"build"/"libsnn_ms.so") as path:
            op = Custom(func=f"{path}:{name}", out_dtype=lambda x: x,
                        out_shape=lambda x: x, bprop=bprop, func_type="aot", reg_info=op_info)
        return op

    @staticmethod
    def get_forward_op(bprop):
        op_info = CustomRegOp() \
            .input(0, "x") \
            .output(0, "out") \
            .dtype_format(DataType.F32_Default, DataType.F32_Default) \
            .target("GPU") \
            .get_op_info()
        return SurrogateBase.get_aot_op("Heaviside", op_info, bprop)

    def construct(self, x):
        return self.op(x)
