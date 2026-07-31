cmake_minimum_required(VERSION 3.20)

# CUDA's nvcc rejects host compilers newer than it supports (CUDA 13.x = g++ 15
# max on Arch). Pin to an older g++ if one is available; users can override
# with -DCMAKE_CUDA_HOST_COMPILER=/path/to/g++.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(_TF_NVCC_HOST_GXX
        NAMES g++-15 g++-14 g++-13
        DOC "Older g++ compatible with the installed CUDA toolkit")
    if(_TF_NVCC_HOST_GXX)
        set(CMAKE_CUDA_HOST_COMPILER "${_TF_NVCC_HOST_GXX}"
            CACHE FILEPATH "Host compiler used by nvcc" FORCE)
        message(STATUS "CUDA host compiler: ${_TF_NVCC_HOST_GXX}")
    endif()
endif()

project(tinyforge LANGUAGES CXX CUDA)

# cuBLAS is used as a *baseline / correctness oracle only* (Phase 2+), never as a
# dependency of our own kernels. CUDAToolkit provides the imported CUDA::cublas.
find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

add_library(tinyforge_ref STATIC
    src/reference/reference.cpp
)
target_include_directories(tinyforge_ref PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(hello_cuda src/hello_cuda.cu)
target_link_libraries(hello_cuda PRIVATE tinyforge_ref)

add_executable(test_setup src/test_setup.cu)
target_link_libraries(test_setup PRIVATE tinyforge_ref)

add_executable(vector_add src/kernels/vector_add.cu)
target_link_libraries(vector_add PRIVATE tinyforge_ref)

add_executable(reduction src/kernels/reduction.cu)
target_link_libraries(reduction PRIVATE tinyforge_ref)

add_executable(gemm_v1_naive src/kernels/gemm_v1_naive.cu)
target_link_libraries(gemm_v1_naive PRIVATE tinyforge_ref CUDA::cublas)

add_executable(gemm_v2_shared src/kernels/gemm_v2_shared.cu)
target_link_libraries(gemm_v2_shared PRIVATE tinyforge_ref CUDA::cublas)

add_executable(gemm_v3_register src/kernels/gemm_v3_register.cu)
target_link_libraries(gemm_v3_register PRIVATE tinyforge_ref CUDA::cublas)

# ── clangd-friendly compile_commands.json ─────────────────────────────────────
# 'clangd_db' runs scripts/fix_compile_commands.py after every build.
# It expands nvcc RSP files and translates flags so clangd understands them.
# Add new .cu/.cpp files to CMakeLists.txt normally — this handles the rest.
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    find_program(_TF_PYTHON python3)
    if(_TF_PYTHON)
        add_custom_target(clangd_db ALL
            COMMAND "${_TF_PYTHON}"
                    "${CMAKE_SOURCE_DIR}/scripts/fix_compile_commands.py"
                    "${CMAKE_BINARY_DIR}/compile_commands.json"
                    "${CMAKE_SOURCE_DIR}/compile_commands.json"
                    --cuda-path /opt/cuda
                    --sm 86
                    --include "${CMAKE_SOURCE_DIR}/include"
            COMMENT "Refreshing clangd compile_commands.json"
            VERBATIM
        )
    endif()
endif()
