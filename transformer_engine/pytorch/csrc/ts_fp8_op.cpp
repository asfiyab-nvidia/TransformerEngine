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
                          int64_t fp8_tensor,
                          int64_t otype
                          )
{
  // otype
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);

  // invoke TE function
  at::Tensor output = cast_to_fp8(input,
                                scale[fp8_tensor],
                                amax[0][fp8_tensor],
                                scale_inv[fp8_tensor],
                                otype_arg
                                );
  return output.clone();
}

at::Tensor cast_from_fp8_ts(const at::Tensor &input,
                              const at::Tensor &scale_inv,
                              int64_t fp8_tensor,
                              int64_t itype,
                              int64_t otype)
{
  // itype
  transformer_engine::DType itype_arg = reverse_map_dtype(itype);
  // otype
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);

  // invoke TE function
  at::Tensor output = cast_from_fp8(input,
                                  scale_inv[fp8_tensor],
                                  itype_arg,
                                  otype_arg
                                );
  return output.clone();
}

at::Tensor fp8_gelu_ts(at::Tensor input,
                      at::Tensor scale,
                      at::Tensor amax,
                      at::Tensor scale_inv,
                      int64_t fp8_tensor,
                      int64_t otype
                      )
{
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);
  at::Tensor output = fp8_gelu(input,
                                scale[fp8_tensor],
                                amax[0][fp8_tensor],
                                scale_inv[fp8_tensor],
                                otype_arg
                                );
  return output;
}

at::Tensor te_gemm_ts(at::Tensor A,
                      at::Tensor A_scale_inverse,
                      int64_t A_type,
                      int64_t transa,
                      at::Tensor B,
                      at::Tensor B_scale_inverse,
                      int64_t B_type,
                      int64_t transb,
                      at::Tensor D,
                      int64_t D_type,
                      at::Tensor bias,
                      at::Tensor pre_gelu_out,
                      int64_t grad,
                      at::Tensor workspace,
                      int64_t workspaceSize,
                      int64_t accumulate,
                      int64_t use_split_accumulator)
{
  // cast inputs to types accepted by te_gemm
  transformer_engine::DType A_type_arg = reverse_map_dtype(A_type);
  bool transa_arg = static_cast<bool>(transa);
  transformer_engine::DType B_type_arg = reverse_map_dtype(B_type);
  bool transb_arg = static_cast<bool>(transb);
  transformer_engine::DType D_type_arg = reverse_map_dtype(D_type);
  bool grad_arg = static_cast<bool>(grad);
  size_t workspaceSize_arg = static_cast<size_t>(workspaceSize);
  bool accumulate_arg = static_cast<bool>(accumulate);
  bool use_split_accumulator_arg = static_cast<bool>(use_split_accumulator);

  te_gemm(A,
          A_scale_inverse,
          A_type_arg,
          transa_arg,
          B,
          B_scale_inverse,
          B_type_arg,
          transb_arg,
          D,
          D_type_arg,
          bias,
          pre_gelu_out,
          grad_arg,
          workspace,
          workspaceSize_arg,
          accumulate_arg,
          use_split_accumulator_arg
          );
  return D;
}

// first arg here defines the namespace where the op is registered
TORCH_LIBRARY(tex_ts, m)
{
  m.def("cast_to_fp8_ts", &cast_to_fp8_ts);
  m.def("cast_from_fp8_ts", &cast_from_fp8_ts);
  m.def("fp8_gelu_ts", &fp8_gelu_ts);
  m.def("te_gemm_ts", &te_gemm_ts);
}
