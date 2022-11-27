import pytest
import torch
from torch import nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions import *
from transformer_engine.pytorch.module import get_workspace
import transformer_engine.pytorch.cpp_extensions as texcpp
import transformer_engine.pytorch.softmax as softmax_defs
import warnings

OPSET = 11


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
                          inp if isinstance(inp, list) else (inp,),
                          fname,
                          verbose=False,
                          opset_version=opset,
                          input_names=["input"],
                          output_names=["output"],
                          do_constant_folding=True,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                          custom_opsets={"tex_ts": 2})


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
    (False, False, False),
    (False, True, True),
])
def test_export_gemm_fp8(use_fp8, use_bias, use_gelu):
    class TestFP8_GEMM(nn.Module):
        def forward(self, inputs):
            inp, weight = inputs
            # inputs to cast_to_fp8
            scale_factor = 1.
            fp8_tensor_inp = tex.FP8FwdTensors.GEMM1_INPUT # Casting to Int happens internally
            fp8_tensor_weight = tex.FP8FwdTensors.GEMM1_WEIGHT
            nb_inp_scales, nb_weight_scales = 1, out_features
            bias_size = nb_weight_scales

            meta_inp = tex.FP8TensorMeta()
            meta_inp.scale = torch.ones(nb_inp_scales, dtype=torch.float32, device="cuda") * scale_factor
            meta_inp.amax_history = torch.zeros(1, nb_inp_scales, dtype=torch.float32, device="cuda")
            meta_inp.scale_inv = torch.ones(nb_inp_scales, dtype=torch.float32, device="cuda") * scale_factor

            meta_weight = tex.FP8TensorMeta()
            meta_weight.scale = torch.ones(nb_weight_scales, dtype=torch.float32, device="cuda") * scale_factor
            meta_weight.amax_history = torch.zeros(1, nb_weight_scales, dtype=torch.float32, device="cuda")
            meta_weight.scale_inv = torch.ones(nb_weight_scales, dtype=torch.float32, device="cuda") * scale_factor

            inp_type = tex.DType.kFloat8E4M3
            weights_type = tex.DType.kFloat8E4M3
            outp_type = torch.float32

            inp_fp8 = cast_to_fp8(
                inp,
                meta_inp,
                fp8_tensor_inp,
                inp_type)

            weight_fp8 = cast_to_fp8(
                weight,
                meta_weight,
                fp8_tensor_weight,
                weights_type)

            # TODO: note that this is FP32 and will not work for now (BF16 is required)
            bias = torch.randn(bias_size, dtype=torch.float32, device="cuda")
            ret = fp8_gemm(
                weight_fp8,
                meta_weight.scale_inv,
                fp8_tensor_weight,
                inp_type,
                inp_fp8,
                meta_inp.scale_inv,
                fp8_tensor_inp,
                weights_type,
                outp_type,
                get_workspace(),
                bias=bias,
                # TODO this should be set to True once we figure out what to do with BF16 bias
                use_bias=False,
                use_split_accumulator=False)

            return ret

    class Test_GEMM(nn.Module):
        def __init__(self, use_bias=False, gelu=False):
            super().__init__()
            self.use_bias = use_bias
            self.gelu = gelu

        def forward(self, inputs):
            inp, weight = inputs
            bias_size = out_features
            outp_type = torch.float32

            bias = torch.randn(bias_size, dtype=torch.float32, device="cuda")
            gelu_input = torch.randn(hidden_size, out_features, dtype=torch.float32, device="cuda")
            # note: due to logic in lines 104:116 and L129 in cpp_extensions.py
            # it appears either bias OR gelu can be activated, not both
            ret, _, _ = gemm(
                weight,
                inp,
                outp_type,
                get_workspace(),

                # test bias
                bias=bias,
                use_bias=self.use_bias,

                # test gelu
                gelu=self.gelu,
                gelu_input=gelu_input,
                grad=False # only True for backward pass
            )

            return ret


    # Set dimensions (these are arbitrary).
    out_features = 128
    hidden_size = 256
    in_features = 64
    inp = torch.randn(hidden_size, in_features, device="cuda")
    weight = torch.randn(out_features, in_features, device="cuda")
    fp8 = "_fp8" if use_fp8 else ""
    bias = "_bias" if use_bias else ""
    gelu = "_gelu" if use_gelu else ""
    fname = f"te.gemm{fp8}{bias}{gelu}.onnx"
    if use_fp8:
        model_fp8 = TestFP8_GEMM()
        use_fp8, use_bias, use_gelu
        do_export(model_fp8, (inp, weight), fname, use_fp8)
    else:
        model_non_fp8 = Test_GEMM(use_bias=use_bias, gelu=use_gelu)
        do_export(model_non_fp8, (inp, weight), fname, use_fp8)


def test_export_layernorm():
    class TestFP8_Layernorm(nn.Module):
        def forward(self, inp):
            # inputs to layernorm_fwd_fp8_ts
            weight = torch.randn(64, 64, dtype=torch.float32, device="cuda")
            bias = torch.randn(64, dtype=torch.float32, device="cuda")
            eps = torch.ones(1, dtype=torch.float32, device="cuda")[0] * 1

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
                fp8_tensor,
                output_type)

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

        def forward(self, inp):
            scale_factor = 1.
            scale_half_tensor = torch.tensor(scale_factor, dtype=torch.float16)
            if self.mask_inp:
                # setting to arbitrary float for testing purposes
                mask = torch.randn(hidden_size, 1, in_features, in_features, device="cuda")
                ret = self.softmax_fn.apply(inp, mask, scale_half_tensor)
            else:
                ret = self.softmax_fn.apply(inp, scale_half_tensor)

            return ret

    # Set dimensions (these are arbitrary).
    in_features = 64
    hidden_size = 256

    if softmax_def == softmax_defs.ScaledUpperTriangMaskedSoftmax:
        inp = torch.randn(hidden_size, in_features, in_features, device="cuda").half()
        fname = "te.ScaledUpperTriangMaskedSoftmax.onnx"
        do_export(Test_Softmax(softmax_def), inp, fname)
    elif softmax_def == softmax_defs.ScaledMaskedSoftmax:
        inp = torch.randn(hidden_size, in_features, in_features, in_features, device="cuda").half()
        fname = "te.ScaledMaskedSoftmax.onnx"
        do_export(Test_Softmax(softmax_def, mask_inp=True), inp, fname)
    elif softmax_def == softmax_defs.ScaledSoftmax:
        inp = torch.randn(hidden_size, in_features, in_features, in_features, device="cuda").half()
        fname = "te.ScaledSoftmax.onnx"
        do_export(Test_Softmax(softmax_def), inp, fname)



@pytest.mark.parametrize("use_fp8", [
    False,
    True,
])
def test_export_linear(use_fp8):
    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256

    inp = torch.randn(hidden_size, in_features, device="cuda")
    fp8 = "_fp8" if use_fp8 else ""
    fname = f"te.linear{fp8}.onnx"
    model = te.Linear(in_features, out_features, bias=True).to(device='cuda')
    do_export(model, inp, fname, use_fp8)


@pytest.mark.parametrize("use_fp8", [
    False,
    True,
])
def test_export_layernorm_linear(use_fp8):
    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256

    inp = torch.randn(in_features, out_features, device="cuda")
    fp8 = "_fp8" if use_fp8 else ""
    fname = f"te.layernorm_linear{fp8}.onnx"
    model = te.LayerNormLinear(hidden_size,
                               3 * hidden_size,
                               bias=True).to(device='cuda')
    do_export(model, inp, fname, use_fp8)


@pytest.mark.parametrize("use_fp8", [
    False,
    True,
])
def test_export_layernorm_mlp(use_fp8):
    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256
    ffn_hidden_size = 256

    inp = torch.randn(in_features, out_features, device="cuda")
    fp8 = "_fp8" if use_fp8 else ""
    fname = f"te.layernorm_mlp{fp8}.onnx"
    model = te.LayerNormMLP(hidden_size,
                            ffn_hidden_size,
                            bias=False).to(device='cuda')
    do_export(model, inp, fname, use_fp8)


# def test_export_core_attention():
#     # Set dimensions (these are arbitrary).
#     in_features = 64
#     out_features = 256
#     kv_channels = 64

#     inp = torch.randn(in_features, out_features, device="cuda")
#     fname = "te.core_attention_fp8.onnx"
#     model = te.transformer.CoreAttention(
#         num_attention_heads=1,
#         kv_channels=kv_channels,
#         attention_dropout=0.5,
#     ).to(device='cuda')
#     do_export(model, inp, fname, use_fp8=True)


# def test_export_multihead_attention():
#     in_features = 64
#     out_features = 256
#     kv_channels = 64

#     inp = torch.randn(in_features, out_features, device="cuda")
#     fname = "te.multihead_attention_fp8.onnx"
#     model = te.transformer.MultiHeadAttention(
#         num_attention_heads=1,
#         kv_channels=kv_channels,
#         attention_dropout=0.5,
#     ).to(device='cuda')
#     do_export(model, inp, fname, use_fp8=True)
