# Use this code with Nvidia's pytorch container which contains
# preinstallted torch and transformer engine.

import torch
from torch import nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_extensions as tex
import transformer_engine.pytorch.cpp_extensions as texcpp
import te_onnx_extensions


OPSET = 14


class TestFP8_Layernorm(nn.Module):
    def __init__(self):
        super().__init__()

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
out_features = 64
hidden_size = 64

# Export to ONNX
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
inp = torch.randn(hidden_size, in_features, device="cuda")
with torch.inference_mode(), te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        model = TestFP8_Layernorm()
        model.cuda().eval()
        torch.onnx.export(model,
                          (inp,),
                          "te.layernorm_fwd_fp8.onnx",
                          verbose=True,
                          opset_version=OPSET,
                          input_names=["input"],
                          output_names=["output"],
                          do_constant_folding=True,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          custom_opsets={"tex_ts": 2})

