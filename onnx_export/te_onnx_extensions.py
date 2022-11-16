#
# ONNX symbolic functions for Transformer Engine
#

import torch
from torch.onnx import symbolic_helper, register_custom_op_symbolic

OPSET = 11


# Asfiya TODO: add scale argument and do a proper export w/ all fields.
@symbolic_helper.parse_args("v", "v", "v", "v", "i", "i")
def onnx_cast_to_fp8(g, input, scale, amax, scale_inv, fp8_tensor, otype):
    return g.op("TRT_FP8QuantizeLinear", input, scale)

@symbolic_helper.parse_args("v", "v", "i", "i", "i")
def onnx_cast_from_fp8(g, input, scale_inv, fp8_tensor, itype, otype):
    return g.op("TRT_FP8DequantizeLinear", input, scale_inv)

register_custom_op_symbolic('tex_ts::cast_to_fp8_ts', onnx_cast_to_fp8, OPSET)
register_custom_op_symbolic('tex_ts::cast_from_fp8_ts', onnx_cast_from_fp8, OPSET)

