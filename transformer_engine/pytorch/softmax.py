# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused scaled masked softmax functions"""
import os
from typing import Callable, Tuple, Union
import torch._C._onnx as _C_onnx
import torch
from torch import nn
from torch.onnx import _type_utils


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        """ScaledUpperTriangMaskedSoftmax fwd"""
        import scaled_upper_triang_masked_softmax_cuda

        scale_t = torch.tensor([scale])
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(
            inputs, scale_t[0]
        )

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(
        ctx, output_grads: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """ScaledUpperTriangMaskedSoftmax bwd"""
        import scaled_upper_triang_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )

        return input_grads, None

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Tensor, scale: float) -> torch.Value:
        def triangular_mask():
            dtype =  _type_utils.JitScalarType.INT64
            ones = torch.onnx.symbolic_opset9.ones_like(g, input, dtype)
            k = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
            mask = g.op("Trilu", ones, k, upper_i=1)
            mask = g.op("Cast", mask, to_i=_C_onnx.TensorProtoDataType.BOOL)
            return mask

        # Captures the logic of function scaled_upper_triang_masked_softmax_warp_forward
        mask = triangular_mask()
        scaled = g.op("Mul", input, scale)
        neg_tenK = g.op("Constant", value_t=torch.tensor(-10000., dtype=torch.float16))
        masked = g.op("Where", mask, neg_tenK, scaled)
        return g.op("Softmax", masked)


class ScaledMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply the mask.
    3. Perform softmax.
    """

    @staticmethod
    def forward(
        ctx, inputs: torch.Tensor, mask: torch.Tensor, scale: float
    ) -> torch.Tensor:
        """ScaledMaskedSoftmax fwd"""
        import scaled_masked_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(
        ctx, output_grads: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """ScaledMaskedSoftmax bwd"""
        import scaled_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None, None

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Value:
        # Captures the logic of function scaled_masked_softmax_warp_forward.
        # output = softmax(mask(input*scale)
        # Computed as:
        #   masked_scaled = (1 - mask)*(input*scale)
        #   softmax_mask = mask * -10000
        #   output = softmax(masked_scaled + softmax_mask)
        scaled = g.op("Mul", input, scale)
        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        inv_mask = g.op("Sub", one, mask)
        # Todo: type is hard coded because softmax uses FP16 or BF16
        neg_tenK = g.op("Constant", value_t=torch.tensor(-10000., dtype=torch.float16))
        softmax_mask = g.op("Mul", mask, neg_tenK)
        masked_scaled = g.op("Mul", inv_mask, scaled)
        masked = g.op("Add", masked_scaled, softmax_mask)
        return g.op("Softmax", masked)
        # An alternative implementation
        # scaled = g.op("Mul", input, scale)
        # neg_tenK = g.op("Constant", value_t=torch.tensor(-10000., dtype=torch.float16))
        # masked = g.op("Where", mask, neg_tenK, scaled)
        # return g.op("Softmax", masked)


class ScaledSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following two operations in sequence
    1. Scale the tensor.
    2. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        """ScaledSoftmax fwd"""
        import scaled_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_softmax_cuda.forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(
        ctx, output_grads: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """ScaledSoftmax bwd"""
        import scaled_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None, None

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Tensor, scale: float) -> torch.Value:
        scaled = g.op("Mul", input, scale)
        return g.op("Softmax", scaled)


class FusedScaleMaskSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        attn_mask_type: attention mask type (pad or causal)
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        attn_mask_type: str,
        mask_func: Callable,
        softmax_in_fp32: bool,
        scale: float,
    ) -> None:
        super().__init__()
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = bool(
            int(os.getenv("NVTE_MASKED_SOFTMAX_FUSION", "1"))
        )
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        assert (
            self.scale is None or softmax_in_fp32
        ), "softmax should be in fp32 when scaled"

    def forward(self, inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """FusedScaleMaskSoftmax fprop"""
        # [b, np, sq, sk]
        assert inp.dim() == 4
        self.input_in_fp16 = inp.dtype == torch.float16
        self.input_in_bf16 = inp.dtype == torch.bfloat16
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16

        if self.is_kernel_available(*inp.size()):
            return self.forward_fused_softmax(inp, mask)
        return self.forward_torch_softmax(inp, mask)

    def is_kernel_available(self, b: int, np: int, sq: int, sk: int) -> bool:
        """Check FusedScaleMaskSoftmax kernel availability based on size"""
        attn_batches = b * np

        if (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.input_in_float16  # input must be fp16
            and 16 < sk <= 4096  # sk must be 16 ~ 2048
            and sq % 4 == 0  # sq must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            if 0 <= sk <= 4096:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)

                if self.attn_mask_type == "causal":
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_fused_softmax(
        self, inp: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Fused masked softmax kernel"""
        b, np, sq, sk = inp.size()
        scale = self.scale if self.scale is not None else 1.0

        if self.attn_mask_type == "causal":
            assert sq == sk, "causal mask is only for self attention"

            # input is 3D tensor (attn_batches, sq, sk)
            inp = inp.view(-1, sq, sk)
            probs = ScaledUpperTriangMaskedSoftmax.apply(inp, scale)
            return probs.view(b, np, sq, sk)
        # input is 4D tensor (b, np, sq, sk)
        if mask is not None:
            return ScaledMaskedSoftmax.apply(inp, mask, scale)
        return ScaledSoftmax.apply(inp, scale)

    def forward_torch_softmax(
        self, inp: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Framework softmax"""
        if self.input_in_float16 and self.softmax_in_fp32:
            inp = inp.float()

        if self.scale is not None:
            inp = inp * self.scale
        mask_output = self.mask_func(inp, mask) if mask is not None else inp
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs

    @staticmethod
    def get_batch_per_block(sq: int, sk: int, b: int, np: int) -> int:
        """Softmax utility"""
        import scaled_masked_softmax_cuda

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)
