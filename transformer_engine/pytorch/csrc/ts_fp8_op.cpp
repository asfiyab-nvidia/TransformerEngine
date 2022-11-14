#include <torch/script.h>
#include "extensions.h"

// how to receive index
at::Tensor cast_to_fp8_ts(const at::Tensor &input,
                          const at::Tensor &scale,
                          const at::Tensor &amax,
                          const at::Tensor &scale_inv
                          )
{
  // otype
  // TE: Typically forward activations and weights require more precision,
  // so E4M3 datatype is best used during forward pass (applies to ONNX export)
  transformer_engine::DType otype = transformer_engine::DType::kFloat8E4M3;

  // this needs to be an input to the wrapper (see FP8FwdTensors)
  int index = 0;

  // invoke TE function
  at::Tensor output = cast_to_fp8(input,
                                scale[index],
                                amax[0][index],
                                scale_inv[index],
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
