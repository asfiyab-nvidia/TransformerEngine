#include <torch/script.h>
#include "extensions.h"

transformer_engine::DType reverse_map_dtype(int64_t dtype)
{
  switch (dtype)
  {
    case static_cast<int64_t>(transformer_engine::DType::kByte):
        return transformer_engine::DType::kByte;
    case static_cast<int64_t>(transformer_engine::DType::kInt32):
        return transformer_engine::DType::kInt32;
    case static_cast<int64_t>(transformer_engine::DType::kFloat32):
        return transformer_engine::DType::kFloat32;
    case static_cast<int64_t>(transformer_engine::DType::kFloat16):
        return transformer_engine::DType::kFloat16;
    case static_cast<int64_t>(transformer_engine::DType::kBFloat16):
        return transformer_engine::DType::kBFloat16;
    case static_cast<int64_t>(transformer_engine::DType::kFloat8E4M3):
        return transformer_engine::DType::kFloat8E4M3;
    case static_cast<int64_t>(transformer_engine::DType::kFloat8E5M2):
        return transformer_engine::DType::kFloat8E5M2;
    default:
        std::cout<<"Invalid input argument\n";
        break;
  }
  return transformer_engine::DType::kFloat8E5M2;
}

at::Tensor cast_to_fp8_ts(const at::Tensor &input,
                          const at::Tensor &scale,
                          const at::Tensor &amax,
                          const at::Tensor &scale_inv,
                          int64_t otype
                          )
{
  // otype
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);

  // invoke TE function
  at::Tensor output = cast_to_fp8(input,
                                scale,
                                amax,
                                scale_inv,
                                otype_arg
                                );
  return output.clone();
}

at::Tensor cast_from_fp8_ts(const at::Tensor &input,
                              const at::Tensor &scale_inv,
                              int64_t itype,
                              int64_t otype)
{
  // itype
  transformer_engine::DType itype_arg = reverse_map_dtype(itype);
  // otype
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);

  // invoke TE function
  at::Tensor output = cast_from_fp8(input,
                                  scale_inv,
                                  itype_arg,
                                  otype_arg
                                );
  return output.clone();
}

// first arg here defines the namespace where the op is registered
TORCH_LIBRARY(tex_ts, m)
{
  m.def("cast_to_fp8_ts", &cast_to_fp8_ts);
  m.def("cast_from_fp8_ts", &cast_from_fp8_ts);
}
