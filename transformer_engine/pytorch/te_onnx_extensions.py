# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
ONNX symbolic functions for Transformer Engine

Warnings of the type pasted below are a known Pytorch issue (https://github.com/pytorch/pytorch/issues/81693):

tests/test_onnx_export.py::test_export_cast_ops[112]
  /opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py:649: UserWarning: The shape inference of trt::TRT_FP8DequantizeLinear type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (Triggered internally at /opt/pytorch/pytorch/torch/csrc/jit/passes/onnx/shape_type_inference.cpp:1880.)
    _C._jit_pass_onnx_graph_shape_type_inference(

"""

import torch
from torch.onnx import symbolic_helper, register_custom_op_symbolic, _type_utils
import transformer_engine_extensions as tex
import torch._C._onnx as _C_onnx

# This file registers custom op symbolic ONNX functions and does not export any symbols.
__all__ = []


# Custom ops spec version
VER = 1


@symbolic_helper.parse_args("v", "v", "v", "v", "i", "i")
def onnx_cast_to_fp8(g, input, scale, amax, scale_inv, fp8_tensor, otype):
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(input)
    if input.type().scalarType() == "Half":
        input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return g.op("trt::TRT_FP8QuantizeLinear", input, scale_inv).setType(input.type().with_dtype(torch.uint8).with_sizes(output_shape))


@symbolic_helper.parse_args("v", "v", "i", "i", "i")
def onnx_cast_from_fp8(g, input, scale_inv, fp8_tensor, itype, otype):
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(input)
    out = g.op("trt::TRT_FP8DequantizeLinear", input, scale_inv).setType(input.type().with_dtype(torch.float32).with_sizes(output_shape))
    if otype == int(tex.DType.kFloat16):
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
    return out


@symbolic_helper.parse_args("v", "v", "v", "v", "i", "i")
def onnx_fp8_gelu(g, input, scale, amax, scale_inv, fp8_tensor, otype):
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(input)
    gelu = torch.onnx.symbolic_opset9.gelu(g, input)
    if input.type().scalarType() == "Half":
        gelu = g.op("Cast", gelu, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    out = g.op("trt::TRT_FP8QuantizeLinear", gelu, scale_inv).setType(input.type().with_dtype(torch.uint8).with_sizes(output_shape))
    return out


@symbolic_helper.parse_args("v", "v", "i", "i", "i",
                             "v", "v", "i", "i", "i",
                             "v", "i", "v", "v", "i",
                             "v", "i", "i", "i")
def onnx_te_gemm(
    g,
    weight,
    weight_scale_inverse,
    weight_fp8_tensor,
    weight_type,
    trans_weight,
    input,
    input_scale_inverse,
    input_fp8_tensor,
    input_type,
    trans_input,
    out,
    out_type,
    bias,
    pre_gelu_out,
    grad,
    workspace,
    workspaceSize,
    accumulate,
    use_split_accumulator):
    is_fp16 = input.type().scalarType() == "Half"
    if input_type == int(tex.DType.kFloat8E4M3):
        input = g.op("trt::TRT_FP8DequantizeLinear", input, input_scale_inverse)

    if weight_type == int(tex.DType.kFloat8E4M3):
        weight = g.op("trt::TRT_FP8DequantizeLinear", weight, weight_scale_inverse)

    output = g.op("Gemm", input, weight, transA_i=trans_input, transB_i=trans_weight)

    empty_tensor_size = [0]
    bias_empty = torch.onnx.symbolic_helper._get_tensor_sizes(bias) == empty_tensor_size
    pre_gelu_out_empty = torch.onnx.symbolic_helper._get_tensor_sizes(pre_gelu_out) == empty_tensor_size
    if pre_gelu_out_empty and not bias_empty:
        if bias.type().scalarType() == "Half":
            output = g.op("Cast", output, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
        output = g.op('Add', output, bias)
    elif not pre_gelu_out_empty and not bias_empty:
        if bias.type().scalarType() == "Half":
            output = g.op("Cast", output, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
        output = g.op('Add', output, bias)
        output = torch.onnx.symbolic_opset9.gelu(g, output)
    # else:
    #     if is_fp16:
    #         output = g.op("Cast", output, to_i=_C_onnx.TensorProtoDataType.FLOAT16)

    return output


@symbolic_helper.parse_args("v", "v", "v", "f", "v", "v", "v",  "i")
def onnx_layernorm_fwd_fp8(g, input, weight, bias, eps, scale, amax, scale_inv, otype):
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
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(input)
    if input.type().scalarType() == "Half":
        ln = g.op("Cast", ln, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    fp8_ln = g.op("trt::TRT_FP8QuantizeLinear", ln, scale_inv).setType(input.type().with_dtype(torch.uint8).with_sizes(output_shape))
    return fp8_ln


@symbolic_helper.parse_args("v", "v", "v", "f", "v", "v", "v",  "i")
def onnx_layernorm_fwd(g, input, weight, bias, eps):
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
    return ln


register_custom_op_symbolic('tex_ts::cast_to_fp8_ts', onnx_cast_to_fp8, VER)
register_custom_op_symbolic('tex_ts::cast_from_fp8_ts', onnx_cast_from_fp8, VER)
register_custom_op_symbolic('tex_ts::fp8_gelu_ts', onnx_fp8_gelu, VER)
register_custom_op_symbolic('tex_ts::te_gemm_ts', onnx_te_gemm, VER)
register_custom_op_symbolic('tex_ts::layernorm_fwd_fp8_inf_ts', onnx_layernorm_fwd_fp8, VER)
register_custom_op_symbolic('tex_ts::layernorm_fwd_inf_ts', onnx_layernorm_fwd, VER)
