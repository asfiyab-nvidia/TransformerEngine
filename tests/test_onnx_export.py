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
    opset: int=OPSET
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
                          input_names=["input"],
                          output_names=["output"],
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
    """Validate the outputs of an ONNX model vs. ONNX Runtime."""
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
            print(f"Detected {len(mismatched_ids)} diverging values.\nShowing first {nb_vals} errors:")
            for loc in mismatched_ids[:nb_vals]:
                print(f"{onnx_output[loc]} -- {torch_output[loc]}")
            raise ValueError(f"Output validation of {fname} failed with {len(mismatches)} errors")


def test_export_cast_ops():
    class TestFP8_QDQ(nn.Module):
        def forward(self, inp):
            scale_factor = 1.
            fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT

            meta = tex.FP8TensorMeta()
            meta.scale = torch.ones(1, dtype=torch.float32, device="cuda") * scale_factor
            meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")
            meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") * scale_factor
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
    do_export(TestFP8_QDQ(), inp, "te.cast_fp8.onnx")


def test_export_gelu_fp8():
    class TestFP8_Gelu(nn.Module):
        def forward(self, inp):
            scale_factor = 1.
            fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT

            meta = tex.FP8TensorMeta()
            meta.scale = torch.ones(1, dtype=torch.float32, device="cuda") * scale_factor
            meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")
            meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") * scale_factor
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


@pytest.mark.parametrize("use_fp8, use_bias, use_gelu", [
    (True, False, False),
    (True, True, False),
    (True, True, True),
    (False, False, False),
    (False, True, False),
    (False, True, True),
])
def test_export_gemm_fp8(use_fp8, use_bias, use_gelu):
    class TestFP8_GEMM(nn.Module):
        def __init__(self, use_bias=False, gelu=False):
            super().__init__()
            self.use_bias = use_bias
            self.gelu = gelu

            scale_factor = 1.
            self.fp8_tensor_inp = tex.FP8FwdTensors.GEMM1_INPUT # Casting to Int happens internally
            self.fp8_tensor_weight = tex.FP8FwdTensors.GEMM1_WEIGHT
            nb_inp_scales, nb_weight_scales = 1, out_features

            self.meta_inp = tex.FP8TensorMeta()
            self.meta_inp.scale = torch.ones(nb_inp_scales, dtype=torch.float32, device="cuda") * scale_factor
            self.meta_inp.amax_history = torch.zeros(1, nb_inp_scales, dtype=torch.float32, device="cuda")
            self.meta_inp.scale_inv = torch.ones(nb_inp_scales, dtype=torch.float32, device="cuda") / scale_factor

            self.meta_weight = tex.FP8TensorMeta()
            self.meta_weight.scale = torch.ones(nb_weight_scales, dtype=torch.float32, device="cuda") * scale_factor
            self.meta_weight.amax_history = torch.zeros(1, nb_weight_scales, dtype=torch.float32, device="cuda")
            self.meta_weight.scale_inv = torch.ones(nb_weight_scales, dtype=torch.float32, device="cuda") / scale_factor

            bias_size = nb_weight_scales
            # TODO: note that this is FP32 and will not work for now (BF16 is required)
            self.bias = torch.randn(bias_size, dtype=torch.float32, device="cuda")
            self.gelu_input = torch.randn(hidden_size, out_features, dtype=torch.float32, device="cuda")

            self.inp_type = tex.DType.kFloat8E4M3
            self.weights_type = tex.DType.kFloat8E4M3
            self.outp_type = torch.float32

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
                # TODO this should be set to True once we figure out what to do with BF16 bias
                use_bias=False,
                use_split_accumulator=False)
            return ret


    class Test_GEMM(nn.Module):
        def __init__(self, use_bias=False, gelu=False):
            super().__init__()
            self.use_bias = use_bias
            self.gelu = gelu

            bias_size = out_features
            self.bias = torch.randn(bias_size, dtype=torch.float32, device="cuda")
            self.gelu_input = torch.randn(hidden_size, out_features, dtype=torch.float32, device="cuda")

        def forward(self, inp, weight):
            outp_type = torch.float32

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
    inp = torch.randn(hidden_size, in_features, device="cuda")
    weight = torch.randn(out_features, in_features, device="cuda")
    fp8_str = "_fp8" if use_fp8 else ""
    bias_str = "_bias" if use_bias else ""
    gelu_str = "_gelu" if use_gelu else ""
    fname = f"te.gemm{fp8_str}{bias_str}{gelu_str}.onnx"
    if use_fp8:
        model = TestFP8_GEMM()
        use_fp8, use_bias, use_gelu
        do_export(model, (inp, weight), fname, use_fp8)
    else:
        model = Test_GEMM(use_bias=use_bias, gelu=use_gelu)
        do_export(model, (inp, weight), fname, use_fp8)
        validate_result(fname, (inp, weight), model, atol=1e-1)


def test_export_layernorm():
    class TestFP8_Layernorm(nn.Module):
        def forward(self, inp):
            # inputs to layernorm_fwd_fp8_ts
            weight = torch.randn(64, 64, dtype=torch.float32, device="cuda")
            bias = torch.randn(64, dtype=torch.float32, device="cuda")
            eps = 1e-4 # An arbitrary small value

            fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT # Casting to Int happens internally

            meta = tex.FP8TensorMeta()
            meta.scale = torch.ones(1, dtype=torch.float32, device="cuda")
            meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")
            meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda")
            output_type = tex.DType.kFloat8E4M3

            ret = texcpp.layernorm_fwd_fp8_inf(
                inp,
                weight,
                bias,
                eps,
                meta,
                output_type)

            ret = cast_from_fp8(ret,
                    meta,
                    fp8_tensor,
                    output_type,
                    tex.DType.kFloat32)
            return ret


    # Set dimensions (these are arbitrary).
    in_features = 64
    hidden_size = 64
    inp = torch.randn(hidden_size, in_features, device="cuda")
    do_export(TestFP8_Layernorm(), inp, "te.layernorm_fwd_fp8.onnx")


@pytest.mark.parametrize("softmax_def", [
    softmax_defs.ScaledUpperTriangMaskedSoftmax,
    softmax_defs.ScaledMaskedSoftmax,
    softmax_defs.ScaledSoftmax,
])
def test_export_softmax(softmax_def):
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
        inp = torch.randn(hidden_size, in_features, in_features, device="cuda").half()
        fname = "te.ScaledUpperTriangMaskedSoftmax.onnx"
        model = Test_Softmax(softmax_def)
    elif softmax_def == softmax_defs.ScaledMaskedSoftmax:
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(hidden_size, 1, in_features, in_features, device="cuda")
        mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)
        # mask = torch.bernoulli(probs).to("cuda", dtype=torch.float16)
        inp = torch.randn(hidden_size, in_features, in_features, in_features, device="cuda").half()
        fname = "te.ScaledMaskedSoftmax.onnx"
        model = Test_Softmax(softmax_def, mask_inp=True)
    elif softmax_def == softmax_defs.ScaledSoftmax:
        inp = torch.randn(hidden_size, in_features, in_features, in_features, device="cuda").half()
        fname = "te.ScaledSoftmax.onnx"
        model = Test_Softmax(softmax_def)
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
@pytest.mark.parametrize("bias", [False,True])
# Todo: handle case of True
@pytest.mark.parametrize("return_bias", [False])
def test_export_linear(use_fp8: bool, bias: bool, return_bias: bool):
    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256

    inp = torch.randn(hidden_size, in_features, device="cuda")
    fp8_str = "_fp8" if use_fp8 else ""
    bias_str = "_bias" if bias else ""
    fname = f"te.linear{fp8_str}{bias_str}.onnx"
    model = te.Linear(in_features, out_features, bias=bias, return_bias=return_bias).to(device='cuda')
    do_export(model, inp, fname, use_fp8)

    if not use_fp8:
        validate_result(fname, inp, model, atol=1e-3)


@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("bias", [False,True])
# Todo: handle case of True
@pytest.mark.parametrize("return_bias", [False])
@pytest.mark.parametrize("return_layernorm_output", [False])
def test_export_layernorm_linear(
    use_fp8: bool,
    bias: bool,
    return_bias: bool,
    return_layernorm_output: bool,
):
    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256

    inp = torch.randn(in_features, out_features, device="cuda")
    fp8_str = "_fp8" if use_fp8 else ""
    bias_str = "_bias" if bias else ""
    fname = f"te.layernorm_linear{fp8_str}{bias_str}.onnx"
    model = te.LayerNormLinear(hidden_size,
                               3 * hidden_size,
                               bias=bias,
                               return_bias=return_bias,
                               return_layernorm_output=return_layernorm_output).to(device='cuda')
    do_export(model, inp, fname, use_fp8)
    if not use_fp8:
        validate_result(fname, inp, model, atol=1e-3)


@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("bias", [False,True])
# Todo: handle case of True
@pytest.mark.parametrize("return_bias", [False])
@pytest.mark.parametrize("return_layernorm_output", [False])
def test_export_layernorm_mlp(
    use_fp8: bool,
    bias: bool,
    return_bias: bool,
    return_layernorm_output: bool,
):
    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256
    ffn_hidden_size = 256

    inp = torch.randn(in_features, out_features, device="cuda", dtype=torch.float32)
    fp8_str = "_fp8" if use_fp8 else ""
    bias_str = "_bias" if bias else ""
    fname = f"te.layernorm_mlp{fp8_str}{bias_str}.onnx"
    model = te.LayerNormMLP(hidden_size,
                            ffn_hidden_size,
                            bias=bias,
                            return_bias=return_bias,
                            return_layernorm_output=return_layernorm_output).to(device='cuda')
    do_export(model, inp, fname, use_fp8)
    if not use_fp8:
        validate_result(fname, inp, model, atol=1e-3)


@pytest.mark.parametrize("attn_mask_type", ["causal", "padding"])
@pytest.mark.parametrize("attention_softmax_in_fp32", [True, False])
@pytest.mark.parametrize("apply_query_key_layer_scaling", [True, False])
def test_export_core_attention(
    attn_mask_type: str,
    attention_softmax_in_fp32: bool,
    apply_query_key_layer_scaling: bool,
):
    # Set dimensions (these are arbitrary).
    kv_channels = 64
    num_attention_heads = 1
    qkv_size = (2048, 4, num_attention_heads, kv_channels)

    query_layer = torch.randn(qkv_size, device="cuda")
    key_layer = torch.randn(qkv_size, device="cuda")
    value_layer = torch.randn(qkv_size, device="cuda")
    attention_mask = None
    inp = (query_layer, key_layer, value_layer, attention_mask)
    sm_prec_str = "_fp32" if attention_softmax_in_fp32 else "_fp16"
    qk_scaling_str = "_qk_scaling" if apply_query_key_layer_scaling else ""
    fname = f"te.core_attention_{attn_mask_type}{sm_prec_str}{qk_scaling_str}.onnx"
    model = te.transformer.CoreAttention(
        num_attention_heads=num_attention_heads,
        kv_channels=kv_channels,
        attention_dropout=0.5,
        attn_mask_type=attn_mask_type,
        attention_softmax_in_fp32=attention_softmax_in_fp32,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
    ).to(device='cuda')
    do_export(model, (query_layer, key_layer, value_layer, attention_mask), fname, use_fp8=True)
    validate_result(fname, inp, model, atol=1e-3)


@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("attn_mask_type", ["causal", "padding"])
@pytest.mark.parametrize("params_dtype", [
    torch.float32,
    #torch.float16 # TODO: handle this
])
# Todo: handle case of True
@pytest.mark.parametrize("input_layernorm", [False])
@pytest.mark.parametrize("return_layernorm_output", [False])
@pytest.mark.parametrize("attention_type", [
    "self",
    #"cross" # TODO: handle this
])
@pytest.mark.parametrize("fuse_qkv_params", [False, True])
def test_export_multihead_attention(
    use_fp8: bool,
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
    hidden_states = torch.randn(sequence_length, batch_size, hidden_size, device="cuda")
    attention_mask = None
    inp = (hidden_states, attention_mask)
    fp8_str = "_fp8" if use_fp8 else ""
    fname = f"te.multihead_attention{fp8_str}.onnx"
    model = te.transformer.MultiHeadAttention(
        *attention_args,
        attn_mask_type=attn_mask_type,
        params_dtype=params_dtype,
        return_layernorm_output=return_layernorm_output,
        input_layernorm=input_layernorm,
        attention_type=attention_type,
        fuse_qkv_params=fuse_qkv_params,
    ).to(device='cuda')
    do_export(model, inp, fname, use_fp8)
    if not use_fp8:
        validate_result(fname, inp, model, atol=1e-3)
