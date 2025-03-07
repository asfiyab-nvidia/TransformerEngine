/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_LOGGING_H_
#define TRANSFORMER_ENGINE_LOGGING_H_

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <nvrtc.h>
#include <string>
#include <stdexcept>

#define NVTE_ERROR(x) \
    do { \
        throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) +            \
                                 " in function " + __func__ + ": " + x);                           \
    } while (false)

#define NVTE_CHECK(x, ...)                                                                         \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            NVTE_ERROR(std::string("Assertion failed: "  #x ". ") + std::string(__VA_ARGS__));     \
        }                                                                                          \
    } while (false)

namespace {

inline void check_cuda_(cudaError_t status) {
    if ( status != cudaSuccess ) {
        NVTE_ERROR("CUDA Error: " + std::string(cudaGetErrorString(status)));
    }
}

inline void check_cublas_(cublasStatus_t status) {
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        NVTE_ERROR("CUBLAS Error: " + std::string(cublasGetStatusString(status)));
    }
}

inline void check_cudnn_(cudnnStatus_t status) {
    if ( status != CUDNN_STATUS_SUCCESS ) {
        std::string message;
        message.reserve(1024);
        message += "CUDNN Error: ";
        message += cudnnGetErrorString(status);
        message += (". "
                    "For more information, enable cuDNN error logging "
                    "by setting CUDNN_LOGERR_DBG=1 and "
                    "CUDNN_LOGDEST_DBG=stderr in the environment.");
        NVTE_ERROR(message);
    }
}

inline void check_nvrtc_(nvrtcResult status) {
    if ( status != NVRTC_SUCCESS ) {
        NVTE_ERROR("NVRTC Error: " + std::string(nvrtcGetErrorString(status)));
    }
}

}  // namespace

#define NVTE_CHECK_CUDA(ans) { check_cuda_(ans); }

#define NVTE_CHECK_CUBLAS(ans) { check_cublas_(ans); }

#define NVTE_CHECK_CUDNN(ans) { check_cudnn_(ans); }

#define NVTE_CHECK_NVRTC(ans) { check_nvrtc_(ans); }

#endif  // TRANSFORMER_ENGINE_LOGGING_H_
