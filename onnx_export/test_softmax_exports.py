# Use this code with Nvidia's pytorch container which contains
# preinstallted torch and transformer engine.

import argparse
import torch
from torch import nn as nn
import transformer_engine.pytorch as te
import transformer_engine.pytorch.softmax as softmax_defs
from transformer_engine.common import recipe

OPSET = 11
# Export to ONNX
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

# Set dimensions (these are arbitrary).
in_features = 64
hidden_size = 256

class Test_Softmax(nn.Module):
    def __init__(self, softmax_function, mask_inp=False):
        super().__init__()
        self.softmax_fn = softmax_function
        self.mask_inp = mask_inp

    def forward(self, inp):
        scale_factor = 1.
        if self.mask_inp:
            # setting to arbitrary float for testing purposes
            mask = torch.randn(inp.shape[0], 1, inp.shape[2], inp.shape[3], device="cuda")
            ret = self.softmax_fn.apply(inp, mask, scale_factor)
        else:
            ret = self.softmax_fn.apply(inp, scale_factor)

        return ret


def export(model, inp, onnx_file_name):
    with torch.no_grad(), te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            model.cuda().eval()
            torch.onnx.export(model,
                            (inp,),
                            onnx_file_name,
                            verbose=True,
                            opset_version=OPSET,
                            input_names=["input"],
                            output_names=["output"],
                            #export_params=True,
                            #do_constant_folding=False,
                            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                            custom_opsets={"tex_ts": 2})


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--triang_masked', action='store_true', help="Export scaled_upper_triang_masked_softmax_cuda model")
parser.add_argument('--masked', action='store_true', help="Export scaled_masked_softmax_cuda model")
parser.add_argument('--regular', action='store_true', help="Export scaled_softmax_cuda model")
parser.add_argument('--all', action='store_true', help="Export all models")
args = parser.parse_args()

if args.all:
    args.triang_masked, args.masked, args.regular = True, True, True
if args.triang_masked:
    model = Test_Softmax(softmax_defs.ScaledUpperTriangMaskedSoftmax)
    inp = torch.randn(hidden_size, in_features, in_features, device="cuda").half()
    export(model, inp, "te.ScaledUpperTriangMaskedSoftmax.onnx")
if args.masked:
    model = Test_Softmax(softmax_defs.ScaledMaskedSoftmax, mask_inp=True)
    inp = torch.randn(hidden_size, in_features, in_features, in_features, device="cuda").half()
    export(model, inp, "te.ScaledMaskedSoftmax.onnx")
if args.regular:
    model = Test_Softmax(softmax_defs.ScaledSoftmax)
    inp = torch.randn(hidden_size, in_features, in_features, in_features, device="cuda").half()
    export(model, inp, "te.ScaledSoftmax.onnx")
