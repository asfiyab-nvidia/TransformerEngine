"""
ONNX symbolic functions for Transformer Engine
"""


import torch
from torch.onnx import symbolic_helper, register_custom_op_symbolic
import transformer_engine_extensions as tex

# This file registers custom op symbolic ONNX functions and does not export any symbols.
__all__ = []

OPSET = 11


@symbolic_helper.parse_args("v", "v", "v", "v", "i", "i")
def onnx_cast_to_fp8(g, input, scale, amax, scale_inv, fp8_tensor, otype):
    return g.op("TRT_FP8QuantizeLinear", input, scale_inv)

@symbolic_helper.parse_args("v", "v", "i", "i", "i")
def onnx_cast_from_fp8(g, input, scale_inv, fp8_tensor, itype, otype):
    return g.op("TRT_FP8DequantizeLinear", input, scale_inv)

@symbolic_helper.parse_args("v", "v", "v", "v", "i", "i")
def onnx_fp8_gelu(g, input, scale, amax, scale_inv, fp8_tensor, otype):
    gelu = torch.onnx.symbolic_opset9.gelu(g, input)
    q = g.op("TRT_FP8QuantizeLinear", gelu, scale)
    return q


@symbolic_helper.parse_args("v", "v", "i", "i", "i",
                             "v", "v", "i", "i", "i",
                             "v", "i",
                             "v", "v", "i", "v",
                             "i", "i", "i")
def onnx_fp8_gemm(g, weight, weight_scale_inverse, weight_fp8_tensor, weight_type, trans_weight,
                      input, input_scale_inverse, input_fp8_tensor, input_type, trans_input,
                      out, out_type,
                      bias,
                      pre_gelu_out,
                      grad,
                      workspace,
                      workspaceSize,
                      accumulate,
                      use_split_accumulator):
    if input_type == int(tex.DType.kFloat8E4M3):
        # add DQ
        input = g.op("TRT_FP8DequantizeLinear", input, input_scale_inverse)

    if weight_type == int(tex.DType.kFloat8E4M3):
        # add DQ
        weight = g.op("TRT_FP8DequantizeLinear", weight, weight_scale_inverse)

    # call gemm op from onnx with trans_weight and trans_input as attributes
    # order specified with Gemm op defined as A x B
    output = g.op("Gemm", input, weight, transA_i=trans_input, transB_i=trans_weight)

    empty_tensor_size = [0]
    bias_empty = torch.onnx.symbolic_helper._get_tensor_sizes(bias) == empty_tensor_size
    pre_gelu_out_empty = torch.onnx.symbolic_helper._get_tensor_sizes(pre_gelu_out) == empty_tensor_size
    if pre_gelu_out_empty and not bias_empty:
        output = g.op('Add', output, bias)
    elif not pre_gelu_out_empty and not bias_empty:
        output = g.op('Add', output, bias)
        output = torch.onnx.symbolic_opset9.gelu(g, output)
    return output


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
    fp8_ln = g.op("TRT_FP8QuantizeLinear", ln, scale_inv)
    return fp8_ln


register_custom_op_symbolic('tex_ts::cast_to_fp8_ts', onnx_cast_to_fp8, OPSET)
register_custom_op_symbolic('tex_ts::cast_from_fp8_ts', onnx_cast_from_fp8, OPSET)
register_custom_op_symbolic('tex_ts::fp8_gelu_ts', onnx_fp8_gelu, OPSET)
register_custom_op_symbolic('tex_ts::te_gemm_ts', onnx_fp8_gemm, OPSET)
register_custom_op_symbolic('tex_ts::layernorm_fwd_fp8_ts', onnx_layernorm_fwd_fp8, OPSET)

