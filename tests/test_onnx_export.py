"""
This file contains tests for exporting TransformerEngine models to ONNX.
"""


import pytest
import torch
from torch import nn as nn
from typing import Union, Tuple
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions import *
from transformer_engine.pytorch.module import get_workspace
import transformer_engine.pytorch.cpp_extensions as texcpp
import transformer_engine.pytorch.softmax as softmax_defs
from transformer_engine.pytorch.utils import get_default_init_method
import onnxruntime as ort
import numpy as np
import warnings


# Opset used in the ONNX files generated by the tests.
OPSET = 15


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
    fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

    with torch.no_grad(), te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe), warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            category=torch.jit.TracerWarning,
            module=r'.*'
        )

        model.cuda().eval()
        torch.onnx.export(model,
                          inp if isinstance(inp, list) or isinstance(inp, tuple) else (inp,),
                          fname,
                          verbose=False,
                          opset_version=opset,
                          input_names=input_names,
                          output_names=output_names,
                          do_constant_folding=True,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                          custom_opsets={"tex_ts": 2})


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def validate_result(
    fname: str,
    inps: Union[Tuple[torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    atol: float,
    max_errors_printed: int=10
):
    """Validate the outputs of an ONNX model vs. ONNX Runtime.

    Use this only to test non-FP8 models because ORT does not support FP8.
    """
    s = ort.InferenceSession(fname)
    inp_dict = {}
    if isinstance(inps, tuple) or isinstance(inps, list):
        for idx, inp in enumerate(inps):
            if inp is None:
                continue
            inp_dict[s.get_inputs()[idx].name] = to_numpy(inp)
    else:
        inp_dict[s.get_inputs()[0].name] = to_numpy(inps)
    onnx_outputs = s.run(None, input_feed=inp_dict)
    torch_outputs = model(*inps if isinstance(inps, tuple) else (inps,))

    if not isinstance(torch_outputs, tuple):
        torch_outputs = (torch_outputs, )

    assert len(onnx_outputs) == len(torch_outputs)
    for onnx_output, torch_output in zip(onnx_outputs, torch_outputs):
        torch_output = to_numpy(torch_output)
        # Compare ORT and PyTorch outputs
        ac = ~np.isclose(onnx_output, torch_output, atol=atol)
        mismatches = ac.nonzero()
        mismatched_ids = [loc for loc in zip(*mismatches)]
        if mismatched_ids:
            # Log some information in case of error.
            print("*" * 100)
            print(onnx_output.shape)
            nb_vals = min(len(mismatched_ids), max_errors_printed)
            print(f"Detected {len(mismatched_ids)} diverging values.\nShowing first {nb_vals} errors (ONNX -- TE):")
            for loc in mismatched_ids[:nb_vals]:
                print(f"{onnx_output[loc]} -- {torch_output[loc]}")
            raise ValueError(f"Output validation of {fname} failed with {len(mismatched_ids)} errors")


def create_meta(scale_factor: float, size: int=1):
    meta = tex.FP8TensorMeta()
    meta.amax_history = torch.zeros(1, size, dtype=torch.float32, device="cuda")
    meta.scale_inv = torch.ones(size, dtype=torch.float32, device="cuda") / scale_factor
    meta.scale = torch.ones(size, dtype=torch.float32, device="cuda") * scale_factor
    return meta


@pytest.mark.parametrize("scale_factor", [448, 112])
def test_export_cast_ops(scale_factor):
    class TestFP8_QDQ(nn.Module):
        def forward(self, inp):
            fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT
            meta = create_meta(scale_factor)
            input_type = tex.DType.kFloat32
            output_type = tex.DType.kFloat8E4M3

            ret = cast_to_fp8(inp,
                              meta,
                              fp8_tensor,
                              output_type)

            ret = cast_from_fp8(ret,
                                meta,
                                fp8_tensor,
                                output_type, # input to cast_to_fp8 is FP8 type
                                input_type)
            return ret
    # Set dimensions (these are arbitrary).
    in_features = 64
    hidden_size = 256
    inp = torch.randn(hidden_size, in_features, device="cuda")
    do_export(TestFP8_QDQ(), inp, f"te.cast_fp8.s_{scale_factor}.onnx")


@pytest.mark.parametrize("scale_factor", [112])
def test_export_gelu_fp8(scale_factor):
    class TestFP8_Gelu(nn.Module):
        def forward(self, inp):
            scale_factor = 1.
            fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT
            meta = create_meta(scale_factor)
            output_type = tex.DType.kFloat8E4M3

            ret = fp8_gelu(inp,
                           meta,
                           fp8_tensor,
                           output_type)
            ret = cast_from_fp8(ret,
                    meta,
                    fp8_tensor,
                    output_type,
                    tex.DType.kFloat32)
            return ret

    # Set dimensions (these are arbitrary).
    in_features = 64
    hidden_size = 256
    inp = torch.randn(hidden_size, in_features, device="cuda")
    do_export(TestFP8_Gelu(), inp, "te.gelu_fp8.onnx")


@pytest.mark.parametrize("scale_factors", [(112, 448,), ])
@pytest.mark.parametrize(
    "precision,     use_fp8, use_bias, use_gelu", [
    (torch.float32, False,   False,    False),
    (torch.float16, False,   False,    False),
    (torch.float32, False,   True,     False),
    (torch.float16, False,   True,     False),
    (torch.float32, False,   True,     True),
    (torch.float16, False,   True,     False),

    # For FP8 GEMM GeLU is not used.
    (torch.float32, True,    False,    False),
    (torch.float16, True,    False,    False),
    # When enabling bias we must use float16 or bfloat16 (because of kernel limitations)
    (torch.float16, True,    True,     False),
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
    high_prec_str = "_fp16" if precision == torch.float16 else "_fp32"
    fname = f"te.gemm{fp8_str}{bias_str}{gelu_str}{high_prec_str}.onnx"
    if use_fp8:
        model = TestFP8_GEMM(precision, use_bias, use_gelu, scale_factors)
        do_export(model, (inp, weight), fname, use_fp8)
    else:
        model = Test_GEMM(precision, use_bias, use_gelu)
        do_export(model, (inp, weight), fname, use_fp8)
        validate_result(fname, (inp, weight), model, atol=1e-1)


@pytest.mark.parametrize("scale_factor", [448])
@pytest.mark.parametrize("precision", [torch.float32, torch.float16])
def test_export_layernorm(scale_factor, precision):
    class TestFP8_Layernorm(nn.Module):
        def forward(self, inp):
            # inputs to layernorm_fwd_fp8_ts
            weight = torch.randn(64, 64, dtype=precision, device="cuda")
            bias = torch.randn(64, dtype=precision, device="cuda")
            eps = 1e-4 # An arbitrary small value

            fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT # Casting to Int happens internally
            meta = create_meta(scale_factor)
            fp8_type = tex.DType.kFloat8E4M3

            ret = texcpp.layernorm_fwd_fp8_inf(
                inp,
                weight,
                bias,
                eps,
                meta,
                fp8_type)

            ret = cast_from_fp8(
                ret,
                meta,
                fp8_tensor,
                fp8_type,
                tex.DType.kFloat32 if precision == torch.float32 else tex.DType.kFloat16)
            return ret

    # Set dimensions (these are arbitrary).
    in_features = 64
    hidden_size = 64
    inp = torch.randn(hidden_size, in_features, device="cuda")
    high_prec_str = "_fp16" if precision == torch.float16 else "_fp32"
    do_export(TestFP8_Layernorm(), inp, f"te.layernorm_fwd_fp8{high_prec_str}.onnx")


@pytest.mark.parametrize("softmax_def", [
    softmax_defs.ScaledUpperTriangMaskedSoftmax,
    softmax_defs.ScaledMaskedSoftmax,
    softmax_defs.ScaledSoftmax,
])
# Softmax kernel only supports FP16 or BF16!
@pytest.mark.parametrize("precision", [torch.float16])
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
    if softmax_def == softmax_defs.ScaledUpperTriangMaskedSoftmax:
        inp = torch.randn(hidden_size, in_features, in_features, device="cuda")
        kernel_str = "te.ScaledUpperTriangMaskedSoftmax"
        model = Test_Softmax(softmax_def)
    elif softmax_def == softmax_defs.ScaledMaskedSoftmax:
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(hidden_size, 1, in_features, in_features, device="cuda")
        mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)
        # mask = torch.bernoulli(probs).to("cuda", dtype=torch.float16)
        inp = torch.randn(hidden_size, in_features, in_features, in_features, device="cuda")
        inp = inp if precision == torch.float32 else inp.half()
        kernel_str = "te.ScaledMaskedSoftmax"
        model = Test_Softmax(softmax_def, mask_inp=True)
    elif softmax_def == softmax_defs.ScaledSoftmax:
        inp = torch.randn(hidden_size, in_features, in_features, in_features, device="cuda")
        kernel_str = "te.ScaledSoftmax"
        model = Test_Softmax(softmax_def)
    inp = inp if precision == torch.float32 else inp.half()
    high_prec_str = "_fp16" if precision == torch.float16 else "_fp32"
    fname = f"{kernel_str}{high_prec_str}.onnx"
    do_export(model, (inp, mask,), fname)

    # TODO: refactor this code (reuse validate_result)
    s = ort.InferenceSession(fname)
    inp_dict = {s.get_inputs()[0].name: to_numpy(inp)}
    if mask is not None:
        inp_dict[s.get_inputs()[1].name] = to_numpy(mask)
    onnx_output = s.run(None, input_feed=inp_dict)[0]
    torch_output = to_numpy(model(inp, mask))

    atol = 1e-3
    ac = ~np.isclose(onnx_output, torch_output, atol=atol)
    mismatches = ac.nonzero()

    mismatched_ids = [loc for loc in zip(*mismatches)]
    if mismatched_ids:
        # Log some information in case of error.
        print("*"*100)
        print(onnx_output.shape)
        max_errors = 20
        nb_vals = min(len(mismatched_ids), max_errors)
        print(f"Detected {len(mismatched_ids)} diverging values.\nShowing first {nb_vals} errors:")
        for loc in mismatched_ids[:nb_vals]:
            print(f"{onnx_output[loc]} -- {torch_output[loc]}")
    assert np.allclose(onnx_output, torch_output, atol=atol)


@pytest.mark.parametrize("use_fp8", [False,True])
# Todo: handle case of True
@pytest.mark.parametrize("return_bias", [False])
@pytest.mark.parametrize(
    "precision,     use_bias",[
    (torch.float32, False),
    (torch.float32, True),
    # Todo: cannot configure FP16 when bias is disabled
    (torch.float16, True),
    #(torch.float16, False),
])
def test_export_linear(
    use_fp8: bool,
    use_bias: bool,
    return_bias: bool,
    precision: torch.dtype):
    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256

    inp = torch.randn(hidden_size, in_features, device="cuda", dtype=precision)
    fp8_str = "_fp8" if use_fp8 else ""
    bias_str = "_bias" if use_bias else ""
    high_prec_str = "_fp16" if precision == torch.float16 else "_fp32"
    fname = f"te.linear{fp8_str}{bias_str}{high_prec_str}.onnx"
    model = te.Linear(
        in_features,
        out_features,
        bias=use_bias,
        return_bias=return_bias,
        params_dtype=precision
    ).to(device='cuda')
    do_export(model, inp, fname, use_fp8)

    if not use_fp8:
        validate_result(fname, inp, model, atol=1e-3)


@pytest.mark.parametrize("use_fp8", [False, True])
# Todo: handle case of True
@pytest.mark.parametrize("return_bias", [False])
@pytest.mark.parametrize("return_layernorm_output", [False])
@pytest.mark.parametrize(
    "precision,     use_bias",[
    (torch.float32, False),
    (torch.float32, True),
    # Todo: cannot configure FP16 when bias is disabled
    (torch.float16, True),
    #(torch.float16, False),
])
def test_export_layernorm_linear(
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
    high_prec_str = "_fp16" if precision == torch.float16 else "_fp32"
    fname = f"te.layernorm_linear{fp8_str}{bias_str}{high_prec_str}.onnx"
    model = te.LayerNormLinear(
        hidden_size,
        3 * hidden_size,
        bias=use_bias,
        return_bias=return_bias,
        return_layernorm_output=return_layernorm_output,
        params_dtype=precision,
    ).to(device='cuda')
    do_export(model, inp, fname, use_fp8)
    if not use_fp8:
        validate_result(fname, inp, model, atol=1e-3)


@pytest.mark.parametrize("use_fp8", [False, True])
#@pytest.mark.parametrize("bias", [False,True])
# Todo: handle case of True
@pytest.mark.parametrize("return_bias", [False])
@pytest.mark.parametrize("return_layernorm_output", [False])
# Todo: cannot handle FP16 for some reason
#@pytest.mark.parametrize("precision", [torch.float32])
@pytest.mark.parametrize(
    "precision,     use_bias",[
    (torch.float32, False),
    (torch.float32, True),
    # Todo: cannot configure FP16 when bias is disabled
    (torch.float16, True),
    #(torch.float16, False),
])
def test_export_layernorm_mlp(
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
    high_prec_str = "_fp16" if precision == torch.float16 else "_fp32"
    fname = f"te.layernorm_mlp{fp8_str}{bias_str}{high_prec_str}.onnx"
    model = te.LayerNormMLP(
        hidden_size,
        ffn_hidden_size,
        bias=use_bias,
        return_bias=return_bias,
        return_layernorm_output=return_layernorm_output,
        params_dtype=precision,
    ).to(device='cuda')
    do_export(model, inp, fname, use_fp8)
    if not use_fp8:
        validate_result(fname, inp, model, atol=1e-3)


test_configs_core_attention = [
    # Torch tests 2 configs
    (True, False, None),
    (True, True, None),
    # TE tests 3 configs
    (False, False, "causal"), # calls ScaledUpperTriangMaskedSoftmax
    (False, True, "padding"), # calls ScaledMaskedSoftmax
    (False, False, "padding"), # calls ScaledSoftmax
]
@pytest.mark.parametrize("use_torch, use_mask, attn_mask_type", test_configs_core_attention)
@pytest.mark.parametrize("attention_softmax_in_fp32", [True, False])
@pytest.mark.parametrize("apply_query_key_layer_scaling", [True, False])
def test_export_core_attention(
    use_torch: bool,
    use_mask: bool,
    attn_mask_type: str,
    attention_softmax_in_fp32: bool,
    apply_query_key_layer_scaling: bool,
):
    if attn_mask_type is None:
        attn_mask_type = 'causal'

    # Set dimensions (these are arbitrary).
    kv_channels = 64
    num_attention_heads = 1
    qkv_size = (2048, 4, num_attention_heads, kv_channels)

    dtype = torch.float16
    if use_torch:
        dtype = torch.float32

    query_layer = torch.randn(qkv_size, dtype=dtype, device="cuda")
    key_layer = torch.randn(qkv_size, dtype=dtype, device="cuda")
    value_layer = torch.randn(qkv_size, dtype=dtype, device="cuda")
    input_names = ["query", "key", "value"]
    attention_mask = None
    if use_mask:
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(qkv_size[1], qkv_size[2], qkv_size[0], qkv_size[0], device="cuda")
        attention_mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)
        input_names.append("attention_mask")
    inp = (query_layer, key_layer, value_layer, attention_mask)

    sm_prec_str = "_fp32" if attention_softmax_in_fp32 else "_fp16"
    qk_scaling_str = "_qk_scaling" if apply_query_key_layer_scaling else ""

    mask_suffix = "_masked" if use_mask else \
                "_upper_trian_masked" if attn_mask_type=="causal" and not use_torch else \
                ""
    torch_suffix = "_torch" if use_torch else ""
    fname = f"te.core_attention{mask_suffix}{torch_suffix}{sm_prec_str}{qk_scaling_str}.onnx"

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


test_configs_multihead_attention = [
    (False, "causal"),  # calls ScaledUpperTriangMaskedSoftmax
    (True, "padding"),  # calls ScaledMaskedSoftmax
    (False, "padding"), # calls ScaledSoftmax
]
@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("use_mask, attn_mask_type", test_configs_multihead_attention)
@pytest.mark.parametrize("params_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("input_layernorm", [True, False])
@pytest.mark.parametrize("return_layernorm_output", [False])
@pytest.mark.parametrize("attention_type", [
    "self",
    #"cross" # TODO: handle this
])
@pytest.mark.parametrize("fuse_qkv_params", [False, True])
def test_export_multihead_attention(
    use_fp8: bool,
    use_mask: bool,
    attn_mask_type: str,
    params_dtype: torch.dtype,
    return_layernorm_output: bool,
    input_layernorm: bool,
    attention_type: str,
    fuse_qkv_params: bool
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
    hidden_states = torch.randn(sequence_length, batch_size, hidden_size, dtype=params_dtype, device="cuda")
    input_names = ["hidden_states"]
    attention_mask = None
    if use_mask and attn_mask_type != "causal":
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(batch_size, 1, sequence_length, sequence_length, device="cuda")
        attention_mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)
        input_names.append("attention_mask")
    inp = (hidden_states, attention_mask)

    fp8_str = "_fp8" if use_fp8 else ""
    dtype_str = "_fp32" if params_dtype == torch.float32 else "_fp16"
    attn_type_str = "_self_attention" if attention_type == "self" else "_cross_attention"
    fuse_qkv_str = "_fused" if fuse_qkv_params else ""
    mask_str = "_masked" if use_mask and attn_mask_type != "causal" else ""
    fname = f"te.multihead_attention{fp8_str}{attn_mask_type}{dtype_str}_{attn_type_str}{fuse_qkv_str}{mask_str}.onnx"

    model = te.transformer.MultiHeadAttention(
        *attention_args,
        attn_mask_type=attn_mask_type,
        params_dtype=params_dtype,
        return_layernorm_output=return_layernorm_output,
        input_layernorm=input_layernorm,
        attention_type=attention_type,
        fuse_qkv_params=fuse_qkv_params,
    ).to(device='cuda')
    do_export(model, inp, fname, use_fp8, input_names=input_names)
    if not use_fp8:
        validate_result(fname, inp, model, atol=1e-3)


@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("use_mask, attn_mask_type", test_configs_multihead_attention)
@pytest.mark.parametrize("output_layernorm", [
    #True, # TO DO: handle this
    False
])
@pytest.mark.parametrize("params_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("fuse_qkv_params", [False, True])
@pytest.mark.parametrize("apply_query_key_layer_scaling", [True, False])
def test_export_transformer_layer(
    use_fp8,
    use_mask,
    attn_mask_type,
    output_layernorm,
    params_dtype,
    fuse_qkv_params,
    apply_query_key_layer_scaling
):
    # Layer configuration
    hidden_size = 64
    sequence_length = 128
    batch_size = 1
    ffn_hidden_size = 256
    num_attention_heads = 4

    input_tensor = torch.rand(sequence_length, batch_size, hidden_size, dtype=params_dtype, device="cuda")
    input_names = ["input"]
    attention_mask = None
    if use_mask and attn_mask_type != "causal":
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(batch_size, 1, sequence_length, sequence_length, device="cuda")
        attention_mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)
        input_names.append("attention_mask")
    inp = (input_tensor, attention_mask)

    fp8 = "_fp8" if use_fp8 else ""
    mask = "_masked" if use_mask and attn_mask_type != "causal" else ""
    fname = f"te.transformer_layer{fp8}{mask}.onnx"
    model = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        self_attn_mask_type=attn_mask_type,
        output_layernorm=output_layernorm,
        params_dtype=params_dtype,
        fuse_qkv_params=fuse_qkv_params,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling).to(device='cuda')
    do_export(model, inp, fname, use_fp8)

    if not use_fp8:
        validate_result(fname, inp, model, atol=1e-3)
