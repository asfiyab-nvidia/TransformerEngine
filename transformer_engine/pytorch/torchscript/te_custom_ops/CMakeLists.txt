# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(torchscript_extension LANGUAGES CUDA CXX)

list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CMAKE_CUDA_FLAGS "-G")
endif()

add_library(torchscript_extension SHARED
                               custom_fp8_ops.cpp)

target_include_directories(torchscript_extension PUBLIC "${PROJECT_SOURCE_DIR}/include")

find_package(CUDAToolkit REQUIRED cublas)

list(APPEND torchscript_extension_LINKER_LIBS CUDA::cublas CUDA::cudart)
target_link_libraries(torchscript_extension PUBLIC ${torchscript_extension})

target_include_directories(torchscript_extension PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
