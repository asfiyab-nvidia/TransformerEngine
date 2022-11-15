# Use this code with Nvidia's pytorch container which contains
# preinstallted torch and transformer engine.

import torch
from torch import nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_extensions as tex
import te_onnx_extensions

OPSET = 11

# Load TS custom operator
torch.ops.load_library("./build/lib.linux-x86_64-3.8/transformer_engine_extensions.cpython-38-x86_64-linux-gnu.so")


class TestFP8_QDQ(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        # inputs to cast_to_fp8
        scale_factor = 1.
        fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT # Casting to Int happens internally

        meta = tex.FP8TensorMeta()
        meta.scale = torch.ones(1, dtype=torch.float32, device="cuda") * scale_factor
        meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")
        meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") * scale_factor
        output_type = tex.DType.kFloat8E4M3

        scale = meta.scale[fp8_tensor]
        amax_history = meta.amax_history[0][fp8_tensor]
        scale_inv = meta.scale_inv[fp8_tensor]
        ret = torch.ops.tex_ts.cast_to_fp8_ts(inp,
                                            meta.scale,
                                            meta.amax_history,
                                            meta.scale_inv,
                                            output_type)

        return ret


# Set dimensions (these are arbitrary).
in_features = 64
out_features = 256
hidden_size = 256

# Export to ONNX
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
inp = torch.randn(hidden_size, in_features, device="cuda")
with torch.no_grad(), te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        model = TestFP8_QDQ()
        model.cuda().eval()
        print(model(inp))
        torch.onnx.export(model,
                          (inp,),
                          "te.cast_fp8.onnx",
                          verbose=True,
                          opset_version=OPSET,
                          input_names=["input_fp32"],
                          output_names=["output_fp8"],
                          #export_params=True,
                          #do_constant_folding=False,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          custom_opsets={"tex_ts": 2})

