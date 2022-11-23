# Use this code with Nvidia's pytorch container which contains
# preinstallted torch and transformer engine.

import torch
from torch import nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions import cast_to_fp8, cast_from_fp8
import te_onnx_extensions

OPSET = 11

# Set dimensions (these are arbitrary).
in_features = 64
out_features = 256
hidden_size = 256

# Export to ONNX
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
inp = torch.randn(hidden_size, in_features, device="cuda")
with torch.no_grad(), te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        model = te.Linear(in_features, out_features, bias=True).to(device='cuda')
        inp = torch.randn(hidden_size, in_features, device="cuda")
        model.cuda().eval()
        print(model(inp))
        torch.onnx.export(model,
                          (inp,),
                          "te.linear.onnx",
                          verbose=True,
                          opset_version=OPSET,
                          input_names=["input"],
                          output_names=["output"],
                          export_params=True,
                          do_constant_folding=True,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          custom_opsets={"tex_ts": 2})
