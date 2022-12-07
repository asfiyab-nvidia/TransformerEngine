
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
from tests.test_onnx_export import do_export

#args
precision = torch.float16
use_mask = False
attn_mask_type = 'causal'
attention_softmax_in_fp32 = True
apply_query_key_layer_scaling = True

# Set dimensions (these are arbitrary).
kv_channels = 64
num_attention_heads = 1
qkv_size = (2048, 4, num_attention_heads, kv_channels)

query_layer = torch.randn(qkv_size, dtype=precision, device="cuda")
key_layer = torch.randn(qkv_size, dtype=precision, device="cuda")
value_layer = torch.randn(qkv_size, dtype=precision, device="cuda")
input_names = ["query", "key", "value"]
attention_mask = None
if use_mask:
    # Generate a random mask with 50% probability for 0 or 1.
    probs = 0.5 * torch.ones(qkv_size[1], qkv_size[2], qkv_size[0], qkv_size[0], device="cuda")
    attention_mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)
    input_names.append("attention_mask")
inp = (query_layer, key_layer, value_layer, attention_mask)

model = te.transformer.CoreAttention(
    num_attention_heads=num_attention_heads,
    kv_channels=kv_channels,
    attention_dropout=0.5,
    attn_mask_type=attn_mask_type,
    attention_softmax_in_fp32=attention_softmax_in_fp32,
    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
).to(device='cuda')

out = model(*inp)
do_export(model,
            inp,
            'abc.onnx',
            input_names=input_names,
            use_fp8=True)