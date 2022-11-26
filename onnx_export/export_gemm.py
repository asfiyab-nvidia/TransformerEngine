# Use this code with Nvidia's pytorch container which contains
# preinstallted torch and transformer engine.

import argparse
import torch
from torch import nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions import cast_to_fp8, fp8_gemm, gemm
from transformer_engine.pytorch.module import get_workspace


OPSET = 11

# Set dimensions (these are arbitrary).
in_features = 64
out_features = 128
hidden_size = 256

# Export to ONNX
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

class TestFP8_GEMM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, weight):
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

    def forward(self, inp, weight):
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



def export(model, onnx_file_name):
    inp = torch.randn(hidden_size, in_features, device="cuda")
    weight = torch.randn(out_features, in_features, device="cuda")
    with torch.inference_mode(), te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            torch.onnx.export(model,
                            (inp, weight),
                            onnx_file_name,
                            verbose=True,
                            opset_version=OPSET,
                            input_names=["input", "weight"],
                            output_names=["output"],
                            #export_params=True,
                            do_constant_folding=True,
                            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                            custom_opsets={"tex_ts": 2})

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--fp8', action='store_true', help="Export FP8 model")
parser.add_argument('--use_bias', action='store_true', help="Export non-FP8 model with bias")
parser.add_argument('--gelu', action='store_true', help="Export non-FP8 model with gelu")

args = parser.parse_args()

if args.fp8:
    model_fp8 = TestFP8_GEMM()
    export(model_fp8, "te.fp8_gemm.onnx")
else:
    model_non_fp8 = Test_GEMM(use_bias=args.use_bias, gelu=args.gelu)
    export(model_non_fp8, "te.non_fp8_gemm.onnx")
