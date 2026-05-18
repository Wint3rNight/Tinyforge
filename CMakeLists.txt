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

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(hello_cuda src/hello_cuda.cu)
