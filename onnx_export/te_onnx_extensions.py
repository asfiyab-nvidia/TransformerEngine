#
# ONNX symbolic functions for Transformer Engine
#

import torch
from torch.onnx import symbolic_helper, register_custom_op_symbolic

OPSET = 11


# Asfiya TODO: add scale argument and do a proper export w/ all fields.
@symbolic_helper.parse_args("v", "v", "v", "v", "i")
def onnx_cast_to_fp8(g, input, scale, amax, scale_inv, otype):
    return g.op("TRT_FP8QuantizeLinear", input, scale)

@symbolic_helper.parse_args("v", "v", "i", "i")
def onnx_cast_from_fp8(g, input, scale_inv, itype, otype):
    return g.op("TRT_FP8DequantizeLinear", input, scale_inv)

@symbolic_helper.parse_args("v", "v", "v", "f", "v", "v", "v",  "i")
def onnx_layernorm_fwd_fp8(g, input, weight, bias, eps, scale, amax, scale_inv, otype):
    return g.op("TRT_FP8QuantizeLinear",
            g.op("LayerNormalization", input, weight, bias, epsilon=eps),
            scale)

register_custom_op_symbolic('tex_ts::cast_to_fp8_ts', onnx_cast_to_fp8, OPSET)
register_custom_op_symbolic('tex_ts::cast_from_fp8_ts', onnx_cast_from_fp8, OPSET)
register_custom_op_symbolic('tex_ts::layernorm_fwd_fp8_ts', onnx_layernorm_fwd_fp8, OPSET)

