#include <torch/script.h>
#include "extensions.h"

// how to receive index
at::Tensor cast_to_fp8_ts(const at::Tensor &input,
                          const at::Tensor &scale
                          )
{
  // otype
  transformer_engine::DType otype = transformer_engine::DType::kFloat8E4M3;

  // amax
  at::Tensor amax = at::zeros({1, 1}, at::device(at::kCUDA));

  // scale_inv
  at::Tensor scale_inv = at::ones_like(scale, at::device(at::kCUDA));

  // invoke TE function
  at::Tensor output = cast_to_fp8(input,
                                scale[0],
                                amax[0][0],
                                scale_inv[0],
                                otype
                                );
  return output.clone();
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
