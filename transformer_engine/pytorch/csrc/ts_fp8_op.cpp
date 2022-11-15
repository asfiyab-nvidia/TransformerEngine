#include <torch/script.h>
#include "extensions.h"

transformer_engine::DType reverse_map_dtype(int64_t dtype)
{
  if (static_cast<int>(transformer_engine::DType::kFloat8E4M3) == dtype)
    return transformer_engine::DType::kFloat8E4M3;
  return transformer_engine::DType::kFloat8E5M2;
}

at::Tensor cast_to_fp8_ts(const at::Tensor &input,
                          const at::Tensor &scale,
                          const at::Tensor &amax,
                          const at::Tensor &scale_inv,
                          int64_t fp8_tensor,
                          int64_t otype
                          )
{
  // otype
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);
  std::cout << "otype " << static_cast<int>(otype_arg) << std::endl;

  // invoke TE function
  at::Tensor output = cast_to_fp8(input,
                                scale[fp8_tensor],
                                amax[0][fp8_tensor],
                                scale_inv[fp8_tensor],
                                otype_arg
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
