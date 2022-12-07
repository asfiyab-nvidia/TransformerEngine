
import pytest
import warnings
import numpy as np
import onnxruntime as ort
import torch
from torch import nn as nn
from typing import Union, Tuple
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions import *
from transformer_engine.pytorch.module import get_workspace
import transformer_engine.pytorch.cpp_extensions as texcpp
import transformer_engine.pytorch.softmax as softmax_defs
from transformer_engine.pytorch.utils import get_default_init_method

class Test_Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, mask):
        scale_factor = 8 # arbitrary value
        ret = softmax_defs.ScaledMaskedSoftmax.apply(inp, mask, scale_factor)
        return ret

model = Test_Softmax()


in_features = 64
hidden_size = 256

# Generate a random mask with 50% probability for 0 or 1.
probs = 0.5 * torch.ones(hidden_size, 1, in_features, in_features, device="cuda")
mask = torch.bernoulli(probs).to("cuda", dtype=torch.half)

input_tensor = torch.randn(hidden_size, in_features, in_features, in_features, device="cuda").half()
inp = (input_tensor, mask)
out = model(*inp)