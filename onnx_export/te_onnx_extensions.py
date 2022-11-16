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

@symbolic_helper.parse_args("v", "v", "i", "i",
                             "v", "v", "i", "i",
                             "v", "i",
                             "v", "v", "i", "v",
                             "i", "i", "i")
def onnx_fp8_gemm(g, weight, weight_scale_inverse, weight_type, trans_weight,
                      input, input_scale_inverse, input_type, trans_input,
                      out, out_type,
                      bias,
                      pre_gelu_out,
                      grad,
                      workspace,
                      workspaceSize,
                      accumulate,
                      use_split_accumulator):
    # put DQ in front of input
    inp_dq = g.op("TRT_FP8DequantizeLinear", input, input_scale_inverse)

    # put DQ in front of weights
    weight_dq = g.op("TRT_FP8DequantizeLinear", weight, weight_scale_inverse)

    # call gemm op from onnx with trans_weight and trans_input as attributes
    # order specified with Gemm op defined as A x B
    return g.op("Gemm", inp_dq, weight_dq, transposeA_i=trans_input, transposeB_i=trans_weight)

register_custom_op_symbolic('tex_ts::cast_to_fp8_ts', onnx_cast_to_fp8, OPSET)
register_custom_op_symbolic('tex_ts::cast_from_fp8_ts', onnx_cast_from_fp8, OPSET)
register_custom_op_symbolic('tex_ts::te_gemm_ts', onnx_fp8_gemm, OPSET)

