#
# ONNX symbolic functions for Transformer Engine
#

import torch
from torch.onnx import symbolic_helper, register_custom_op_symbolic
import transformer_engine_extensions as tex

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
    if input_type == int(tex.DType.kFloat8E4M3):
        # add DQ
        input = g.op("TRT_FP8DequantizeLinear", input, input_scale_inverse)

    if weight_type == int(tex.DType.kFloat8E4M3):
        # add DQ
        weight = g.op("TRT_FP8DequantizeLinear", weight, weight_scale_inverse)

    # call gemm op from onnx with trans_weight and trans_input as attributes
    # order specified with Gemm op defined as A x B
    out = g.op("Gemm", input, weight, transposeA_i=trans_input, transposeB_i=trans_weight)

    #TO DO: better way to check empty tensors
    if (pre_gelu_out.node().kind() == "onnx::Constant" and bias.node().kind() != "onnx::Constant"):
        out = g.op('Add', out, bias)

    #TO DO: ONNX doesn't have gelu
    if (pre_gelu_out.node().kind() != "onnx::Constant" and bias.node().kind() != "onnx::Constant"):
        # out = g.op('TRT_Gelu', out, pre_gelu_out)
        out = torch.onnx.symbolic_opset9.gelu(g, out)
    return out

register_custom_op_symbolic('tex_ts::cast_to_fp8_ts', onnx_cast_to_fp8, OPSET)
register_custom_op_symbolic('tex_ts::cast_from_fp8_ts', onnx_cast_from_fp8, OPSET)
register_custom_op_symbolic('tex_ts::te_gemm_ts', onnx_fp8_gemm, OPSET)

