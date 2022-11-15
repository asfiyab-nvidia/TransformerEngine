#
# ONNX symbolic functions for Transformer Engine
#

import torch
from torch.onnx import symbolic_helper, register_custom_op_symbolic

OPSET = 11


# Asfiya TODO: add scale argument and do a proper export w/ all fields.
@symbolic_helper.parse_args("v", "v", "v", "v", "i")
def onnx_cast_to_fp8(g, input, scale, amax, scale_inv, fp8_tensor):
    return g.op("TRT_FP8DequantizeLinear", g.op("TRT_FP8QuantizeLinear", input, scale), scale)

@symbolic_helper.parse_args("v")
def onnx_cast_from_fp8(g, input):
    return g.op("trt::cast_from_fp8", input)


register_custom_op_symbolic('tex_ts::cast_to_fp8_ts', onnx_cast_to_fp8, OPSET)
register_custom_op_symbolic('tex_ts::cast_from_fp8_ts', onnx_cast_from_fp8, OPSET)

