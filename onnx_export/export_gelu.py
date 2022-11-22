# Use this code with Nvidia's pytorch container which contains
# preinstallted torch and transformer engine.

import torch
from torch import nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions import fp8_gelu
import te_onnx_extensions

OPSET = 11

class TestFP8_Gelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        # inputs to gelu_fp8
        scale_factor = 1.
        fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT # Casting to Int happens internally

        meta = tex.FP8TensorMeta()
        meta.scale = torch.ones(1, dtype=torch.float32, device="cuda") * scale_factor
        meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")
        meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") * scale_factor
        output_type = tex.DType.kFloat8E4M3

        ret = fp8_gelu(inp,
                        meta,
                        fp8_tensor,
                        output_type)

        return ret


# Set dimensions (these are arbitrary).
in_features = 64
hidden_size = 256

# Export to ONNX
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
inp = torch.randn(hidden_size, in_features, device="cuda")
with torch.no_grad(), te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        model = TestFP8_Gelu()
        model.cuda().eval()
        print(model(inp))
        torch.onnx.export(model,
                          (inp,),
                          "te.gelu_fp8.onnx",
                          verbose=True,
                          opset_version=OPSET,
                          input_names=["input_fp32"],
                          output_names=["output_fp8"],
                          #export_params=True,
                          #do_constant_folding=False,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          custom_opsets={"tex_ts": 2})

