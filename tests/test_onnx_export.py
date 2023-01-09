# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
This file contains tests for exporting TransformerEngine models to ONNX.
"""

import os
import pytest
import warnings
import numpy as np
import math
import onnxruntime as ort
import torch
from torch import nn as nn
from typing import Union, Tuple, List
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions import *
from transformer_engine.pytorch.module import get_workspace
import transformer_engine.pytorch.cpp_extensions as texcpp
import transformer_engine.pytorch.softmax as softmax_defs
from transformer_engine.pytorch.utils import get_default_init_method


# Directory where generated ONNX test models are stored.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_FILES_DIR = os.path.join(TESTS_DIR, "./gen_onnx_models")

# Shared library implementing custom FP8 Q/DQ operators for ONNX Runtime (ORT).
ORT_CUSTOM_OPS_LIB = "./tests/libcustom_ort_fp8_qdq_ops.so"

# ScaledUpperTriangMaskedSoftmax is exported via ONNX::Trilu which was introduced in opset 14.
TRILU_OPSET = 14
# Opset used in the ONNX files generated by the tests.
OPSET = 15
assert OPSET >= TRILU_OPSET


def create_fp8_recipe():
    return recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)


def do_export(
    model: torch.nn.Module,
    inp: torch.Tensor,
    fname: str,
    use_fp8: bool=True,
    opset: int=OPSET,
    input_names: list=["input"],
    output_names: list=["output"],
):
    """Export to ONNX"""
    fp8_recipe = create_fp8_recipe()

    with torch.inference_mode(), te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe), warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            category=torch.jit.TracerWarning,
            module=r'.*'
        )

        model.cuda().eval()
        os.makedirs(ONNX_FILES_DIR, exist_ok=True)
        fname = os.path.join(ONNX_FILES_DIR, fname)
        torch.onnx.export(model,
                          inp if isinstance(inp, list) or isinstance(inp, tuple) else (inp,),
                          fname,
                          verbose=False,
                          opset_version=opset,
                          input_names=input_names,
                          output_names=output_names,
                          do_constant_folding=True,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)


def to_numpy(tensor):
    return tensor.cpu().numpy()


def set_layer_scale(module: torch.nn.Module, scales: List[float]):
    module.fp8_init()
    num_fp8_tensors = len(scales)
    scale = torch.ones(num_fp8_tensors, dtype=torch.float32, device="cuda")
    scale_inv = torch.ones(num_fp8_tensors, dtype=torch.float32, device="cuda")
    amax_history_len = module.fp8_meta["recipe"].amax_history_len
    amax_history = torch.zeros(amax_history_len, num_fp8_tensors, dtype=torch.float32, device="cuda")
    for i, s in enumerate(scales):
       scale[i] *= s
       scale_inv[i] /= s
    module.fp8_meta["scaling_fwd"].scale = scale
    module.fp8_meta["scaling_fwd"].scale_inv = scale_inv
    module.fp8_meta["scaling_fwd"].amax_history = amax_history


def te_infer(model: torch.nn.Module, inps: Union[Tuple[torch.tensor], torch.tensor], is_fp8: bool):
    """Transformer Engine forward prpoagtation.

    Return results after copying to the CPU and converting to numpy.
    """
    fp8_recipe = create_fp8_recipe()
    with torch.inference_mode(), te.fp8_autocast(enabled=is_fp8, fp8_recipe=fp8_recipe), warnings.catch_warnings():
        te_outputs = model(*inps if isinstance(inps, tuple) else (inps,))
        if not isinstance(te_outputs, tuple):
            te_outputs = (te_outputs,)
        te_outputs_np = [to_numpy(te_output) for te_output in te_outputs]
        return te_outputs_np


def validate_result(
    fname: str,
    inps: Union[Tuple[torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    atol: float=1.e-8, # np.isclose default atol
    rtol: float=1.e-5, # np.isclose default rtol
    max_errors_printed: int=10,
    is_fp8: bool=False,
):
    """Validate the outputs of an ONNX model vs. ONNX Runtime."""

    def create_ort_session(fname: str, is_fp8: bool):
        def load_custom_ops(session_opts: ort.SessionOptions):
            """For FP8 validation with ORT we need to load our custom FP8 Q/DQ extension."""
            if not os.path.exists(ORT_CUSTOM_OPS_LIB):
                raise FileNotFoundError(f"Unable to find {ORT_CUSTOM_OPS_LIB}")
            session_opts.register_custom_ops_library(ORT_CUSTOM_OPS_LIB)
            print("registered custom FP8 Q/DQ ops!")

        """Create an ONNX Runtime session for validation."""
        if is_fp8:
            sess_options = ort.SessionOptions()
            load_custom_ops(sess_options)
            # Model loading successfully indicates that the custom op node could be resolved successfully
            s = ort.InferenceSession(fname, sess_options=sess_options)
        else:
            s = ort.InferenceSession(fname)
        return s

    def create_ort_input_dict(session, inps):
        inp_dict = {}
        if isinstance(inps, tuple) or isinstance(inps, list):
            nonetype_inputs = 0
            for idx, inp in enumerate(inps):
                if inp is None:
                    nonetype_inputs += 1
                    continue
                inp_dict[session.get_inputs()[idx - nonetype_inputs].name] = to_numpy(inp)
        else:
            inp_dict[session.get_inputs()[0].name] = to_numpy(inps)
        return inp_dict

    # Run ORT session and TE model.
    fname = os.path.join(ONNX_FILES_DIR, fname)
    ort_s = create_ort_session(fname, is_fp8)
    onnx_outputs = ort_s.run(None, input_feed=create_ort_input_dict(ort_s, inps))
    te_outputs = te_infer(model, inps, is_fp8)

    # Compare ORT and TE outputs.
    assert len(onnx_outputs) == len(te_outputs)
    for onnx_output, te_output in zip(onnx_outputs, te_outputs):

        # Compare ORT and PyTorch outputs.
        # np.isclose: abs(a - b) <= (atol + rtol * abs(b))
        ac = ~np.isclose(onnx_output, te_output, atol=atol, rtol=rtol)

        mismatches = ac.nonzero()
        mismatched_ids = [loc for loc in zip(*mismatches)]
        if mismatched_ids:
            # Log some information in case of error.
            print("*" * 100)
            print(onnx_output.shape)
            nb_vals = min(len(mismatched_ids), max_errors_printed)
            print(f"Detected {len(mismatched_ids)} diverging values.\nShowing first {nb_vals} errors (ONNX -- TE):")
            abs_err = abs(onnx_output - te_output)
            for loc in mismatched_ids[:nb_vals]:
                ref = te_output[loc]
                print(f"{onnx_output[loc]} -- {te_output[loc]} err={abs_err[loc]} > {atol + rtol * abs(ref)}")
            raise ValueError(f"Output validation of {fname} failed with {len(mismatched_ids)} errors")


def create_meta(scale_factor: float, size: int=1):
    meta = tex.FP8TensorMeta()
    meta.amax_history = torch.zeros(1, size, dtype=torch.float32, device="cuda")
    meta.scale_inv = torch.ones(size, dtype=torch.float32, device="cuda") / scale_factor
    meta.scale = torch.ones(size, dtype=torch.float32, device="cuda") * scale_factor
    return meta


def dtype2str(dtype: torch.dtype):
    return {
        torch.float32: "_fp32",
        torch.float16: "_fp16",
        torch.bfloat16: "_bf16",
    }[dtype]


def as_te_type(dtype: torch.dtype):
    return {
        torch.float32: tex.DType.kFloat32,
        torch.float16: tex.DType.kFloat16,
        torch.bfloat16: tex.DType.kBFloat16,
    }[dtype]


def get_attn_mask_str(use_mask, attn_mask_type):
    # See FusedScaleMaskSoftmax::forward_fused_softmax for logic behind names.
    if attn_mask_type is None:
        return "_mask" if use_mask else "_no-mask"
    attn_mask_str = "_padding-no-mask"
    attn_mask_str = "_causal-mask" if attn_mask_type == "causal" else attn_mask_str
    attn_mask_str = "_padding-mask" if use_mask and attn_mask_type == "padding" else attn_mask_str
    return attn_mask_str


@pytest.mark.parametrize("scale_factor, atol", [
    (1, 1e-7),
    (224, 1e-7)
])
@pytest.mark.parametrize("precision", [torch.float32, torch.float16])
def test_export_cast_ops(scale_factor: float, atol: float, precision: torch.dtype):
    class TestFP8_QDQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.fp8_tensor = 0
            self.meta = create_meta(scale_factor)
            self.highprec_type = as_te_type(precision)
            self.fp8_type = tex.DType.kFloat8E4M3

        def forward(self, inp):
            ret = cast_to_fp8(
                inp,
                self.meta,
                self.fp8_tensor,
                self.fp8_type)

            ret = cast_from_fp8(
                ret,
                self.meta,
                self.fp8_tensor,
                self.fp8_type,
                self.highprec_type)
            return ret

    # Set dimensions (these are arbitrary).
    in_features = 64
    hidden_size = 256
    inp = torch.randn(hidden_size, in_features, device="cuda", dtype=precision)
    high_prec_str = dtype2str(precision)
    fname = f"te.cast_fp8_{scale_factor}{high_prec_str}.onnx"
    model = TestFP8_QDQ()
    do_export(model, inp, fname)
    validate_result(fname, inp, model, atol=atol, is_fp8=True)


@pytest.mark.parametrize("scale_factor", [448])
@pytest.mark.parametrize(
    "precision,     atol", [
    [torch.float32, 1e-7],
    [torch.float16, 2e-3]
])
def test_export_gelu_fp8(scale_factor: float, precision: torch.dtype, atol: float):
    class TestFP8_Gelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.fp8_tensor = 0
            self.meta = create_meta(scale_factor)
            self.highprec_type = as_te_type(precision)
            self.fp8_type = tex.DType.kFloat8E4M3

        def forward(self, inp):
            ret = fp8_gelu(
                inp,
                self.meta,
                self.fp8_tensor,
                self.fp8_type)
            ret = cast_from_fp8(
                ret,
                self.meta,
                self.fp8_tensor,
                self.fp8_type,
                self.highprec_type)
            return ret

    # Set dimensions (these are arbitrary).
    in_features = 64
    hidden_size = 256
    inp = torch.randn(hidden_size, in_features, device="cuda", dtype=precision)
    high_prec_str = dtype2str(precision)
    fname = f"te.gelu_fp8_{scale_factor}{high_prec_str}.onnx"
    model = TestFP8_Gelu()
    do_export(model, inp, fname)
    validate_result(fname, inp, model, rtol=1e-1, atol=atol, is_fp8=True)


@pytest.mark.parametrize("scale_factors",
    [(224, 224,),
])
@pytest.mark.parametrize(
    "precision,     use_fp8, use_bias, use_gelu", [
    (torch.float32, False,   False,    False),
    (torch.float16, False,   False,    False),
    (torch.float32, False,   True,     False),
    (torch.float16, False,   True,     False),
    (torch.float32, False,   True,     True),
    (torch.float16, False,   True,     True),

    # For FP8 GEMM GeLU is not used.
    (torch.float32, True,    False,    False),
    (torch.float16, True,    False,    False),
    # When enabling bias we must use float16 or bfloat16 (because of kernel limitations)
    (torch.float16, True,    True,     False),
    (torch.bfloat16, True,   True,     False),
])
def test_export_gemm(
    precision, # Precision of inputs, weights, output and bias
    use_fp8,
    use_bias,
    use_gelu,
    scale_factors
):
    class TestFP8_GEMM(nn.Module):
        def __init__(self, precision, use_bias, gelu, scale_factors):
            super().__init__()
            self.use_bias = use_bias
            self.gelu = gelu
            self.precision = precision

            self.fp8_tensor_inp = tex.FP8FwdTensors.GEMM1_INPUT
            self.fp8_tensor_weight = tex.FP8FwdTensors.GEMM1_WEIGHT
            nb_inp_scales, nb_weight_scales = 1, out_features
            act_scale_factor, weight_scale_factor = scale_factors
            self.meta_inp = create_meta(act_scale_factor, nb_inp_scales)
            self.meta_weight = create_meta(weight_scale_factor, nb_weight_scales)

            bias_size = nb_weight_scales
            self.bias = torch.randn(bias_size, dtype=precision, device="cuda")
            self.gelu_input = torch.randn(hidden_size, out_features, dtype=precision, device="cuda")

            self.inp_type = tex.DType.kFloat8E4M3
            self.weights_type = tex.DType.kFloat8E4M3
            self.outp_type = precision

        def forward(self, inp, weight):
            inp_fp8 = cast_to_fp8(
                inp,
                self.meta_inp,
                self.fp8_tensor_inp,
                self.inp_type)

            weight_fp8 = cast_to_fp8(
                weight,
                self.meta_weight,
                self.fp8_tensor_weight,
                self.weights_type)

            ret = fp8_gemm(
                weight_fp8,
                self.meta_weight.scale_inv,
                self.fp8_tensor_weight,
                self.inp_type,
                inp_fp8,
                self.meta_inp.scale_inv,
                self.fp8_tensor_inp,
                self.weights_type,
                self.outp_type,
                get_workspace(),
                bias=self.bias,
                use_bias=self.use_bias,
                fp32_output=(self.precision==torch.float32),
                use_split_accumulator=False)
            return ret

    class Test_GEMM(nn.Module):
        def __init__(self, precision, use_bias=False, gelu=False):
            super().__init__()
            self.use_bias = use_bias
            self.gelu = gelu
            self.precision = precision
            bias_size = out_features
            self.bias = torch.randn(bias_size, dtype=precision, device="cuda")
            self.gelu_input = torch.randn(hidden_size, out_features, dtype=precision, device="cuda")

        def forward(self, inp, weight):
            outp_type = self.precision

            # note: due to logic in lines 104:116 and L129 in cpp_extensions.py
            # it appears either bias OR gelu can be activated, not both
            ret, _, _ = gemm(
                weight,
                inp,
                outp_type,
                get_workspace(),

                # test bias
                bias=self.bias,
                use_bias=self.use_bias,

                # test gelu
                gelu=self.gelu,
                gelu_input=self.gelu_input,
                grad=False # only True for backward pass
            )
            return ret

    # If gelu is applied then bias must be added, as defined by TE kernel.
    if use_gelu: assert use_bias
    # Set dimensions (these are arbitrary).
    out_features = 128
    hidden_size = 256
    in_features = 64
    inp = torch.randn(hidden_size, in_features, dtype=precision, device="cuda")
    weight = torch.randn(out_features, in_features, dtype=precision, device="cuda")
    fp8_str = "_fp8" if use_fp8 else ""
    bias_str = "_bias" if use_bias else ""
    gelu_str = "_gelu" if use_gelu else ""
    high_prec_str = dtype2str(precision)
    fname = f"te.gemm{fp8_str}{bias_str}{gelu_str}{high_prec_str}.onnx"
    if use_fp8:
        model = TestFP8_GEMM(precision, use_bias, use_gelu, scale_factors)
        do_export(model, (inp, weight), fname, use_fp8)
        if precision not in (torch.bfloat16, torch.float16):
            validate_result(fname, (inp, weight), model, rtol=1e-2, atol=1e-2, is_fp8=True)
    else:
        model = Test_GEMM(precision, use_bias, use_gelu)
        do_export(model, (inp, weight), fname, use_fp8)
        validate_result(fname, (inp, weight), model, rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("scale_factor", [448, 112])
@pytest.mark.parametrize("precision", [torch.float32, torch.float16])
def test_export_layernorm(
    use_fp8: bool,
    scale_factor: float,
    precision: torch.dtype
):
    # Set dimensions (these are arbitrary).
    inp_shape = [64, 32]

    class Test_Layernorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            normalized_shape = torch.Size(inp.shape[1:])
            self.weight = torch.randn(*normalized_shape, dtype=precision, device="cuda")
            self.bias = torch.zeros(*normalized_shape, dtype=precision, device="cuda")
            self.eps = 1e-6 # An arbitrary small value

        def forward(self, inp):
            ret = texcpp.layernorm_fwd_inf(
                inp,
                self.weight,
                self.bias,
                self.eps)
            return ret

    class TestFP8_Layernorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            normalized_shape = torch.Size(inp.shape[1:])
            self.weight = torch.randn(*normalized_shape, dtype=precision, device="cuda")
            self.bias = torch.zeros(*normalized_shape, dtype=precision, device="cuda")
            self.eps = 1e-6 # An arbitrary small value

            self.fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT
            self.meta = create_meta(scale_factor)
            self.fp8_type = tex.DType.kFloat8E4M3

        def forward(self, inp):
            ret = texcpp.layernorm_fwd_fp8_inf(
                inp,
                self.weight,
                self.bias,
                self.eps,
                self.meta,
                self.fp8_tensor,
                self.fp8_type)

            ret = cast_from_fp8(
                ret,
                self.meta,
                self.fp8_tensor,
                self.fp8_type,
                tex.DType.kFloat32 if precision == torch.float32 else tex.DType.kFloat16)
            return ret

    inp = torch.randn(*inp_shape, device="cuda", dtype=precision)
    model = TestFP8_Layernorm() if use_fp8 else Test_Layernorm()
    high_prec_str = dtype2str(precision)
    fp8_str = f"_fp8-{scale_factor}" if use_fp8 else ""
    fname = f"te.layernorm{fp8_str}{high_prec_str}.onnx"
    do_export(model, inp, fname)
    if precision not in (torch.bfloat16, ):
        # TODO: FP32 has a small threshold (1e-5)
        validate_result(fname, inp, model, atol=1e-3, is_fp8=use_fp8)


@pytest.mark.parametrize("softmax_def", [
    softmax_defs.ScaledUpperTriangMaskedSoftmax,
    softmax_defs.ScaledMaskedSoftmax,
    softmax_defs.ScaledSoftmax,
])
# Softmax kernel only supports FP16 or BF16!
@pytest.mark.parametrize("precision", [torch.float16, torch.bfloat16])
def test_export_softmax(softmax_def, precision):
    class Test_Softmax(nn.Module):
        def __init__(self, softmax_function, mask_inp=False):
            super().__init__()
            self.softmax_fn = softmax_function
            self.mask_inp = mask_inp

        def forward(self, inp, mask):
            scale_factor = 8 # arbitrary value
            if self.mask_inp:
                ret = self.softmax_fn.apply(inp, mask, scale_factor)
            else:
                ret = self.softmax_fn.apply(inp, scale_factor)
            return ret

    # Set dimensions (these are arbitrary).
    in_features = 64
    hidden_size = 256
    mask = None
    input_names = ["input"]
    inp_shape = [hidden_size, in_features, in_features, in_features]
    if softmax_def == softmax_defs.ScaledUpperTriangMaskedSoftmax:
        inp_shape = [hidden_size, in_features, in_features]
        kernel_str = "ScaledUpperTriangMaskedSoftmax"
        model = Test_Softmax(softmax_def)
    elif softmax_def == softmax_defs.ScaledMaskedSoftmax:
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(hidden_size, 1, in_features, in_features, device="cuda", dtype=precision)
        mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)
        input_names.append("mask")
        kernel_str = "ScaledMaskedSoftmax"
        model = Test_Softmax(softmax_def, mask_inp=True)
    elif softmax_def == softmax_defs.ScaledSoftmax:
        kernel_str = "ScaledSoftmax"
        model = Test_Softmax(softmax_def)
    input_tensor = torch.randn(*inp_shape, device="cuda")
    input_tensor = input_tensor.to(torch.bfloat16) if precision == torch.bfloat16 else input_tensor.half()
    high_prec_str = dtype2str(precision)
    fname = f"{kernel_str}{high_prec_str}.onnx"
    inp = (input_tensor, mask)
    do_export(model, inp, fname, input_names=input_names)
    if precision != torch.bfloat16:
        validate_result(fname, inp, model, atol=1e-3)


@pytest.mark.parametrize("scale_factors", [[448, 448]])
@pytest.mark.parametrize("use_fp8", [False, True])
# Returning the bias is a TE fusion optimization we don't care about.
@pytest.mark.parametrize("return_bias", [False])
@pytest.mark.parametrize(
    "precision,     use_bias",[
    (torch.float32, False),
    (torch.float32, True),
    (torch.float16, False),
    (torch.float16, True),
    # Todo: cannot configure BF16 when bias is disabled (ORT issue?)
    (torch.bfloat16, False),
    # Todo: cannot configure BF16 when bias is enabled (ORT issue?)
    # (torch.bfloat16, True),
])
def test_export_linear(
    scale_factors: List[float],
    use_fp8: bool,
    use_bias: bool,
    return_bias: bool,
    precision: torch.dtype
):
    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256

    class Test_Linear(nn.Module):
        def __init__(self,
                in_features,
                out_features,
                use_bias,
                return_bias,
                precision
            ):
            super().__init__()
            self.linear = te.Linear(
                in_features,
                out_features,
                bias=use_bias,
                return_bias=return_bias,
                params_dtype=precision
            )

        def forward(self, inp):
            ret = self.linear(inp)
            return ret


    inp = torch.randn(hidden_size, in_features, device="cuda", dtype=precision)
    fp8_str = "_fp8" if use_fp8 else ""
    bias_str = "_bias" if use_bias else ""
    high_prec_str = dtype2str(precision)
    fname = f"te.linear{fp8_str}{bias_str}{high_prec_str}.onnx"
    with te.fp8_autocast(enabled=use_fp8, fp8_recipe=create_fp8_recipe()):
        model = Test_Linear(
            in_features,
            out_features,
            use_bias,
            return_bias,
            precision
        ).to(device='cuda')
        if use_fp8:
            set_layer_scale(model.linear, scale_factors)
        do_export(model, inp, fname, use_fp8)

        if precision in (torch.bfloat16, ):
            return
        if not use_fp8:
            validate_result(fname, inp, model, atol=5e-4)
        else:
            validate_result(fname, inp, model, atol=5e-4, is_fp8=use_fp8)


@pytest.mark.parametrize("scale_factors", [[448, 448]])
@pytest.mark.parametrize("use_fp8", [False, True])
# Returning the bias is a TE fusion optimization we don't care about.
@pytest.mark.parametrize("return_bias", [False])
@pytest.mark.parametrize("return_layernorm_output", [False])
@pytest.mark.parametrize(
    "precision,     use_bias",[
    (torch.float32, False),
    (torch.float32, True),
    (torch.float16, True),
    (torch.float16, False),
])
def test_export_layernorm_linear(
    scale_factors: List[float],
    use_fp8: bool,
    use_bias: bool,
    return_bias: bool,
    return_layernorm_output: bool,
    precision: torch.dtype
):
    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256

    inp = torch.randn(in_features, out_features, device="cuda", dtype=precision)
    fp8_str = "_fp8" if use_fp8 else ""
    bias_str = "_bias" if use_bias else ""
    high_prec_str = dtype2str(precision)
    fname = f"te.layernorm_linear{fp8_str}{bias_str}{high_prec_str}.onnx"
    with te.fp8_autocast(enabled=use_fp8, fp8_recipe=create_fp8_recipe()):
        model = te.LayerNormLinear(
            hidden_size,
            3 * hidden_size,
            bias=use_bias,
            return_bias=return_bias,
            return_layernorm_output=return_layernorm_output,
            params_dtype=precision,
        ).to(device='cuda')
        if use_fp8:
            set_layer_scale(model, scale_factors)
        do_export(model, inp, fname, use_fp8)
        if not use_fp8:
            validate_result(fname, inp, model, atol=1e-3)
        elif precision not in (torch.bfloat16,):
            validate_result(fname, inp, model, atol=1e-3, is_fp8=use_fp8)


@pytest.mark.parametrize("scale_factors", [[224, 224, 448, 448]])
@pytest.mark.parametrize("use_fp8", [False, True])
# Returning the bias is a TE fusion optimization we don't care about.
@pytest.mark.parametrize("return_bias", [False])
@pytest.mark.parametrize("return_layernorm_output", [False])
@pytest.mark.parametrize(
    "precision,     use_bias",[
    (torch.float32, False),
    (torch.float32, True),
    (torch.float16, True),
    (torch.float16, False),
])
def test_export_layernorm_mlp(
    scale_factors: List[float],
    use_fp8: bool,
    use_bias: bool,
    return_bias: bool,
    return_layernorm_output: bool,
    precision: torch.dtype
):
    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256
    ffn_hidden_size = 256

    inp = torch.randn(in_features, out_features, device="cuda", dtype=precision)
    fp8_str = "_fp8" if use_fp8 else ""
    bias_str = "_bias" if use_bias else ""
    high_prec_str = dtype2str(precision)
    fname = f"te.layernorm_mlp{fp8_str}{bias_str}{high_prec_str}.onnx"
    with te.fp8_autocast(enabled=use_fp8, fp8_recipe=create_fp8_recipe()):
        model = te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            bias=use_bias,
            return_bias=return_bias,
            return_layernorm_output=return_layernorm_output,
            params_dtype=precision,
        ).to(device='cuda')
        if use_fp8:
            set_layer_scale(model, scale_factors)
        do_export(model, inp, fname, use_fp8)
        if not use_fp8:
            validate_result(fname, inp, model, atol=5e-4)
        else:
            validate_result(fname, inp, model, atol=7e-3, is_fp8=use_fp8)


@pytest.mark.parametrize(
    "precision,     use_mask, attn_mask_type", [
    (torch.float32, False,    None),      # calls forward_torch_softmax
    (torch.float32, True,     None),      # calls forward_torch_softmax
    (torch.float16, False,    "causal"),  # calls ScaledUpperTriangMaskedSoftmax
    (torch.float16, True,     "padding"), # calls ScaledMaskedSoftmax
    (torch.float16, False,    "padding"), # calls ScaledSoftmax
])
@pytest.mark.parametrize("attention_softmax_in_fp32",
    [True, False])
@pytest.mark.parametrize("apply_query_key_layer_scaling",
    [True, False])
def test_export_core_attention(
    precision: torch.dtype,
    use_mask: bool,
    attn_mask_type: str,
    attention_softmax_in_fp32: bool,
    apply_query_key_layer_scaling: bool,
):
    # Set dimensions (these are arbitrary).
    kv_channels = 64
    num_attention_heads = 1
    qkv_size = (2048, 4, num_attention_heads, kv_channels)

    query_layer = torch.randn(qkv_size, dtype=precision, device="cuda")
    key_layer = torch.randn(qkv_size, dtype=precision, device="cuda")
    value_layer = torch.randn(qkv_size, dtype=precision, device="cuda")
    input_names = ["query", "key", "value"]
    attention_mask = None
    if use_mask:
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(qkv_size[1], qkv_size[2], qkv_size[0], qkv_size[0], device="cuda", dtype=precision)
        attention_mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)
        input_names.append("attention_mask")
    inp = (query_layer, key_layer, value_layer, attention_mask)

    sm_prec_str = "_sm-fp32" if attention_softmax_in_fp32 else "_sm-fp16"
    qk_scaling_str = "_qk-scaling" if apply_query_key_layer_scaling else ""
    mask_str = get_attn_mask_str(use_mask, attn_mask_type)
    high_prec_str = dtype2str(precision)
    fname = f"te.core_attention{mask_str}{qk_scaling_str}{sm_prec_str}{high_prec_str}.onnx"

    if attn_mask_type is None:
        attn_mask_type = 'causal'
    model = te.transformer.CoreAttention(
        num_attention_heads=num_attention_heads,
        kv_channels=kv_channels,
        attention_dropout=0.5,
        attn_mask_type=attn_mask_type,
        attention_softmax_in_fp32=attention_softmax_in_fp32,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
    ).to(device='cuda')
    do_export(model,
            inp,
            fname,
            input_names=input_names,
            use_fp8=True)
    validate_result(fname, inp, model, atol=1e-2)


def set_mha_scales(module,
    scale_factor_qkv: List[float]=[448, 448],
    scale_factor_query: List[float]=[112, 112],
    scale_factor_kv: List[float]=[224, 224],
    scale_factor_proj: List[float]=[448, 448]
):
    if module.attention_type == "self":
        if module.input_layernorm:
            # LayernormLinear layer scale init
            set_layer_scale(module.layernorm_qkv, scale_factor_qkv)
        else:
            # Linear layer scale init
            set_layer_scale(module.qkv, scale_factor_qkv)
    else:
        if module.input_layernorm:
            # LayernormLinear layer scale init
            set_layer_scale(module.layernorm_query, scale_factor_query)
        else:
            # Linear layer scale init
            set_layer_scale(module.query_layer, scale_factor_query)

        # Linear layer scale init
        set_layer_scale(module.key_value, scale_factor_kv)

    # Linear layer scale init
    set_layer_scale(module.proj, scale_factor_proj)

test_configs_multihead_attention = [
    #"use_mask, attn_mask_type"
    (False,    "causal"),  # calls ScaledUpperTriangMaskedSoftmax
    (True,     "padding"), # calls ScaledMaskedSoftmax
    (False,    "padding"), # calls ScaledSoftmax
]
test_configs_attention_type = [
    #"input_layernorm, attention_type, fuse_qkv_params"
    (True,             "self",         True),
    (False,            "self",         True),
    (True,             "self",         False),
    (False,            "self",         False),
    # disabled because query_bias (reqd for cross attention) is defined when fuse_qkv_params is False
    # (True,           "cross",        True),
    # (False,          "cross",        True),
    (True,             "cross",        False),
    # disabled because TypeError: cannot assign 'transformer_engine.pytorch.module.Linear'
    # as parameter 'query' (torch.nn.Parameter or None expected)
    # (False,          "cross",        False),
]
@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("use_mask, attn_mask_type", test_configs_multihead_attention)
@pytest.mark.parametrize("precision", [torch.float32, torch.float16])
@pytest.mark.parametrize("return_layernorm_output", [False])
@pytest.mark.parametrize("input_layernorm, attention_type, fuse_qkv_params", test_configs_attention_type)
@pytest.mark.parametrize("scale_factor_qkv", [[448, 448]])
@pytest.mark.parametrize("scale_factor_query", [[112, 112]])
@pytest.mark.parametrize("scale_factor_kv", [[224, 224]])
@pytest.mark.parametrize("scale_factor_proj", [[448, 448]])
def test_export_multihead_attention(
    use_fp8: bool,
    use_mask: bool,
    attn_mask_type: str,
    precision: torch.dtype,
    return_layernorm_output: bool,
    input_layernorm: bool,
    attention_type: str,
    fuse_qkv_params: bool,
    scale_factor_qkv: List[float],
    scale_factor_query: List[float],
    scale_factor_kv: List[float],
    scale_factor_proj: List[float],
):
    hidden_size = 256
    sequence_length = 128
    batch_size = 4
    num_attention_heads = 32
    kv_channels = 8
    attention_dropout = 0.1
    layernorm_epsilon = 1e-5
    init_method = output_layer_init_method = get_default_init_method()
    attention_args = (
        hidden_size,
        num_attention_heads,
        kv_channels,
        attention_dropout,
        layernorm_epsilon,
        init_method,
        output_layer_init_method,
    )
    hidden_states = torch.randn(sequence_length, batch_size, hidden_size, dtype=precision, device="cuda")

    attention_mask = None
    if use_mask and attn_mask_type != "causal":
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(batch_size, 1, sequence_length, sequence_length, device="cuda", dtype=precision)
        attention_mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)

    encoder_output = None
    if attention_type == "cross":
        encoder_output = torch.randn(sequence_length, batch_size, hidden_size, dtype=precision, device="cuda")
    inp = (hidden_states, attention_mask, encoder_output)
    input_names = ["hidden_states", "attention_mask", "encoder_output"]

    fp8_str = "_fp8" if use_fp8 else ""
    dtype_str = dtype2str(precision)
    attn_type_str = "_self-attention" if attention_type == "self" else "_cross-attention"
    fuse_qkv_str = "_fused-qkv" if fuse_qkv_params else ""
    attn_mask_str = get_attn_mask_str(use_mask, attn_mask_type)
    input_ln_str = "_input-ln" if input_layernorm else ""
    fname = f"te.multihead_attention{fp8_str}{attn_mask_str}{attn_type_str}{input_ln_str}{fuse_qkv_str}{dtype_str}.onnx"

    with te.fp8_autocast(enabled=use_fp8, fp8_recipe=create_fp8_recipe()):
        model = te.transformer.MultiHeadAttention(
            *attention_args,
            attn_mask_type=attn_mask_type,
            params_dtype=precision,
            return_layernorm_output=return_layernorm_output,
            input_layernorm=input_layernorm,
            attention_type=attention_type,
            fuse_qkv_params=fuse_qkv_params,
        ).to(device='cuda')
        if use_fp8:
            set_mha_scales(model,
                scale_factor_qkv,
                scale_factor_query,
                scale_factor_kv,
                scale_factor_proj)

        do_export(model, inp, fname, use_fp8, input_names=input_names)
        if not use_fp8:
            validate_result(fname, inp, model, atol=1e-3)
        elif precision != torch.float16:
            validate_result(fname, inp, model, atol=5e-3, is_fp8=use_fp8)

def set_transformer_layer_scales(module,
    scales_self_attn: list,
    scales_inter_attn: list,
    scales_layernorm_mlp: list=[224, 224, 448, 448]):
    # set mha scales
    set_mha_scales(module.self_attention, *scales_self_attn)
    if module.layer_type == "decoder":
        set_mha_scales(module.inter_attention, *scales_inter_attn)
    # set layernorm mlp scales
    set_layer_scale(module.layernorm_mlp, scales_layernorm_mlp)

@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("use_mask, attn_mask_type", test_configs_multihead_attention)
@pytest.mark.parametrize("output_layernorm", [
    #True, # TO DO: handle this
    False
])
@pytest.mark.parametrize("precision", [torch.float32, torch.float16])
@pytest.mark.parametrize("fuse_qkv_params", [False, True])
@pytest.mark.parametrize("apply_query_key_layer_scaling", [True, False])
@pytest.mark.parametrize("scale_factor_qkv", [[448, 448]])
@pytest.mark.parametrize("scale_factor_query", [[112, 112]])
@pytest.mark.parametrize("scale_factor_kv", [[224, 224]])
@pytest.mark.parametrize("scale_factor_proj", [[448, 448]])
@pytest.mark.parametrize("scale_factor_layernorm_mlp", [[224, 224, 448, 448]])
def test_export_transformer_layer(
    use_fp8: bool,
    use_mask: bool,
    attn_mask_type: str,
    output_layernorm: bool,
    precision: torch.dtype,
    fuse_qkv_params: bool,
    apply_query_key_layer_scaling: bool,
    scale_factor_qkv: List[float],
    scale_factor_query: List[float],
    scale_factor_kv: List[float],
    scale_factor_proj: List[float],
    scale_factor_layernorm_mlp: List[float],
):
    # Layer configuration
    hidden_size = 64
    sequence_length = 128
    batch_size = 1
    ffn_hidden_size = 256
    num_attention_heads = 4

    input_tensor = torch.rand(sequence_length, batch_size, hidden_size, dtype=precision, device="cuda")
    input_names = ["input"]
    attention_mask = None
    if use_mask and attn_mask_type != "causal":
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(batch_size, 1, sequence_length, sequence_length, device="cuda", dtype=precision)
        attention_mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)
        input_names.append("attention_mask")
    inp = (input_tensor, attention_mask)

    fp8_str = "_fp8" if use_fp8 else ""
    fuse_qkv_params_str = "_fused-qkv" if fuse_qkv_params else ""
    qk_scaling_str = "_qk-scaling" if apply_query_key_layer_scaling else ""
    high_prec_str = dtype2str(precision)
    attn_mask_str = get_attn_mask_str(use_mask, attn_mask_type)
    fname = f"te.transformer_layer{fp8_str}{attn_mask_str}{fuse_qkv_params_str}{qk_scaling_str}{high_prec_str}.onnx"

    with te.fp8_autocast(enabled=use_fp8, fp8_recipe=create_fp8_recipe()):
        model = te.TransformerLayer(
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            self_attn_mask_type=attn_mask_type,
            output_layernorm=output_layernorm,
            params_dtype=precision,
            fuse_qkv_params=fuse_qkv_params,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling).to(device='cuda')
        if use_fp8:
            mha_scales = [
                scale_factor_qkv,
                scale_factor_query,
                scale_factor_kv,
                scale_factor_proj
            ]
            set_transformer_layer_scales(model,
                scales_self_attn=mha_scales,
                scales_inter_attn=mha_scales,
                scales_layernorm_mlp=scale_factor_layernorm_mlp)

        do_export(model, inp, fname, use_fp8)
        if not use_fp8:
            validate_result(fname, inp, model, atol=1e-3)
        elif precision != torch.float16:
            validate_result(fname, inp, model, atol=1e-2, is_fp8=use_fp8)
