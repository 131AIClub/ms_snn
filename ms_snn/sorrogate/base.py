import mindspore.nn as nn
from mindspore.ops import Custom, CustomRegOp, DataType
from mindspore import dtype, jit
from importlib.resources import files, as_file


class SurrogateBase(nn.Cell):
    def __init__(self):
        super(SurrogateBase, self).__init__()

    @staticmethod
    def get_aot_op(name: str, op_info, out_dtype, out_shape, bprop=None):
        pkg = files("ms_snn")
        with as_file(pkg / "sorrogate"/"cuda"/"libms_snn.so") as path:
            # 这里不用管什么类型
            op = Custom(func=f"{path}:{name}", out_dtype=out_dtype,
                        out_shape=out_shape, bprop=bprop, func_type="aot", reg_info=op_info)
        return op

    @staticmethod
    def get_forward_op(bprop):
        op_info = CustomRegOp() \
            .input(0, "x") \
            .output(0, "out") \
            .dtype_format(DataType.F32_Default, DataType.F32_Default) \
            .target("GPU") \
            .get_op_info()
        return SurrogateBase.get_aot_op("Heaviside", op_info, lambda x: x, lambda x: x, bprop)

    @jit
    def construct(self, x):
        return self.op(x)
