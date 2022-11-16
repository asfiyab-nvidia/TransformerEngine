# Use this code with Nvidia's pytorch container which contains
# preinstallted torch and transformer engine.

import torch
from torch import nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions import cast_to_fp8, cast_from_fp8, fp8_gemm
from transformer_engine.pytorch.module import get_workspace
import te_onnx_extensions

OPSET = 11

class TestFP8_GEMM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, weight):
        # inputs to cast_to_fp8
        scale_factor = 1.
        fp8_tensor_inp = tex.FP8FwdTensors.GEMM1_INPUT # Casting to Int happens internally
        fp8_tensor_weight = tex.FP8FwdTensors.GEMM1_WEIGHT
        inp_meta_size, weight_meta_size = 1, 2

        meta_inp = tex.FP8TensorMeta()
        meta_inp.scale = torch.ones(inp_meta_size, dtype=torch.float32, device="cuda") * scale_factor
        meta_inp.amax_history = torch.zeros(1, inp_meta_size, dtype=torch.float32, device="cuda")
        meta_inp.scale_inv = torch.ones(inp_meta_size, dtype=torch.float32, device="cuda") * scale_factor

        meta_weight = tex.FP8TensorMeta()
        meta_weight.scale = torch.ones(weight_meta_size, dtype=torch.float32, device="cuda") * scale_factor
        meta_weight.amax_history = torch.zeros(1, weight_meta_size, dtype=torch.float32, device="cuda")
        meta_weight.scale_inv = torch.ones(weight_meta_size, dtype=torch.float32, device="cuda") * scale_factor
    
        fp8_type = tex.DType.kFloat8E4M3
        fp32_type = torch.float32

        inp_fp8 = cast_to_fp8(inp,
                            meta_inp,
                            fp8_tensor_inp,
                            fp8_type)

        weight_fp8 = cast_to_fp8(weight,
                                meta_weight,
                                fp8_tensor_weight,
                                fp8_type)

        scale_inv_weights = torch.ones(weight_meta_size, dtype=torch.float32, device="cuda")[fp8_tensor_weight]
        scale_inv_inp = torch.ones(inp_meta_size, dtype=torch.float32, device="cuda")[fp8_tensor_inp]
        bias = torch.randn(128, dtype=torch.bfloat16, device="cuda")
        ret = fp8_gemm(weight_fp8,
                        scale_inv_weights,
                        fp8_type,
                        inp_fp8,
                        scale_inv_inp,
                        fp8_type,
                        fp32_type,
                        get_workspace(),
                        bias=bias,
                        use_bias=True,
                        use_split_accumulator=False
                        )

        return ret


# Set dimensions (these are arbitrary).
in_features = 64
out_features = 256
hidden_size = 256

# Export to ONNX
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
inp = torch.randn(hidden_size, in_features, device="cuda")
weight = torch.randn(128, in_features, device="cuda")
with torch.no_grad(), te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        model = TestFP8_GEMM()
        model.cuda().eval()
        output = model(inp, weight)
        print(output)
        print(output.shape)
        torch.onnx.export(model,
                          (inp, weight),
                          "te.fp8_gemm.onnx",
                          verbose=True,
                          opset_version=OPSET,
                          input_names=["input", "weight"],
                          output_names=["output"],
                          #export_params=True,
                          #do_constant_folding=False,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          custom_opsets={"tex_ts": 2})

