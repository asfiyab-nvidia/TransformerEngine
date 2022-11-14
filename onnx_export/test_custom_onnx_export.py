# Use this code with Nvidia's pytorch container which contains
# preinstallted torch and transformer engine.

import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from torch import nn as nn
import te_onnx_extensions

OPSET = 11

# Load TS custom operator
torch.ops.load_library("./build/lib.linux-x86_64-3.8/transformer_engine_extensions.cpython-38-x86_64-linux-gnu.so")


class TestFP8_QDQ(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        scale = 1.
        scale = torch.ones(1, dtype=torch.float32, device="cuda") * scale
        # use the custom op loaded above
        ret = torch.ops.tex_ts.cast_to_fp8_ts(inp, scale)
        ret = torch.ops.tex_ts.cast_from_fp8_ts(ret)
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
        print(model)
        torch.onnx.export(model,
                          (inp,),
                          "te.cast_fp8.onnx",
                          verbose=True,
                          opset_version=OPSET,
                          #export_params=True,
                          #do_constant_folding=False,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          custom_opsets={"tex": 2})

