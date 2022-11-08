#include <torch/script.h>
#include "extensions.h"


torch::Tensor cast_to_fp8_ts(const at::Tensor &input,
                          const at::Tensor &scale)
{
  // Should invoke: texcpp.cast_to_fp8(inp, meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_type)
  // Scale input will be used to initialize arguments needed by texcpp.cast_to_fp8

  // otype
  transformer_engine::DType otype = transformer_engine::DType::kFloat8E4M3;

  // perform necessary conversions to scale?

  // amax
  at::Tensor amax = at::zeros((1, 1), at::CUDA(GetATenDType(otype)));

  // scale_inv
  at::Tensor scale_inv = at::ones_like(scale, at::CUDA(GetATenDType(otype)));

  // at::Tensor temp = cast_to_fp8(input,
  //                               scale,
  //                               amax,
  //                               scale_inv,
  //                               otype);
  return input.clone();
}

torch::Tensor cast_from_fp8_ts(torch::Tensor X)
{
  // Should invoke: texcpp.cast_to_fp8(inp, meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_type)
  // should it? Or should it just call cast_from_fp8() from the c++ def? why do we go python->c++->python->c++
  return X.clone();
}

// first arg here defines the namespace where the op is registered
TORCH_LIBRARY(tex_ts, m)
{
  m.def("cast_to_fp8_ts", &cast_to_fp8_ts);
  m.def("cast_from_fp8_ts", &cast_from_fp8_ts);
}

