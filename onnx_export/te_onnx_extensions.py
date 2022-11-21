#
# ONNX symbolic functions for Transformer Engine
#

import torch
from torch.onnx import symbolic_helper, register_custom_op_symbolic
import transformer_engine_extensions as tex

OPSET = 14


# Asfiya TODO: add scale argument and do a proper export w/ all fields.
@symbolic_helper.parse_args("v", "v", "v", "v", "i")
def onnx_cast_to_fp8(g, input, scale, amax, scale_inv, otype):
    return g.op("TRT_FP8QuantizeLinear", input, scale)


@symbolic_helper.parse_args("v", "v", "i", "i")
def onnx_cast_from_fp8(g, input, scale_inv, itype, otype):
    return g.op("TRT_FP8DequantizeLinear", input, scale_inv)


@symbolic_helper.parse_args("v", "v", "v", "f", "v", "v", "v",  "i")
def onnx_layernorm_fwd_fp8(g, input, weight, bias, eps, scale, amax, scale_inv, otype):#, normalized_shape):
    normalized_shape = torch.onnx.symbolic_helper._get_tensor_sizes(input)
    if normalized_shape is None:
        ndim = torch.onnx.symbolic_helper._get_tensor_rank(input)
        assert ndim is not None
        normalized_shape = list(range(0, ndim))
    # Normalization axis = 0, so normalized_shape uses all dims except dim = 0
    normalized_shape = normalized_shape[1:]

    ln = torch.onnx.symbolic_opset9.layer_norm(
        g,
        input,
        normalized_shape,
        weight,
        bias,
        eps,
        False # cudnn_enable (not relevant)
    )
    fp8_ln = g.op("TRT_FP8QuantizeLinear", ln, scale)
    return fp8_ln


register_custom_op_symbolic('tex_ts::cast_to_fp8_ts', onnx_cast_to_fp8, OPSET)
register_custom_op_symbolic('tex_ts::cast_from_fp8_ts', onnx_cast_from_fp8, OPSET)
register_custom_op_symbolic('tex_ts::layernorm_fwd_fp8_ts', onnx_layernorm_fwd_fp8, OPSET)


