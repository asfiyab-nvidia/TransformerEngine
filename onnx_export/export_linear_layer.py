# Use this code with Nvidia's pytorch container which contains
# preinstallted torch and transformer engine.

import argparse
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

fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
inp = torch.randn(hidden_size, in_features, device="cuda")


def export(onnx_name, fp8=False):
        with torch.no_grad(), te.fp8_autocast(enabled=fp8, fp8_recipe=fp8_recipe):
                model = te.Linear(in_features, out_features, bias=True).to(device='cuda')
                model.cuda().eval()
                # print(model(inp))
                torch.onnx.export(model,
                                (inp,),
                                onnx_name,
                                verbose=True,
                                opset_version=OPSET,
                                input_names=["input"],
                                output_names=["output"],
                                export_params=True,
                                do_constant_folding=True,
                                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                                custom_opsets={"tex_ts": 2})

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--fp8', action='store_true', help="Export FP8 model")

args = parser.parse_args()

if args.fp8:
    export("te.linear_fp8.onnx", fp8=True)
else:
    export("te.linear_non_fp8.onnx")

