cmake_minimum_required(VERSION 3.24)
project(fastllm LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CheckLanguage)
check_language(CUDA)
# Canonical-path bitwise determinism across TUs and engines: implicit
# multiply-add contraction is compiler- and context-dependent (aarch64 gcc
# contracts per-TU at -O3; nvcc fmads by default), which broke run_serial vs
# CPU-pool vs GPU bitwise equality on thor (P1.0). Contraction is disabled
# globally; FAST kernels keep their throughput via EXPLICIT fmaf()/vfmaq
# intrinsics, which these flags do not touch.
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>")
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  string(APPEND CMAKE_CUDA_FLAGS " --fmad=false")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 110)
  endif()
  add_compile_definitions(FASTLLM_CUDA=1)
  message(STATUS "fastllm: CUDA enabled (arch ${CMAKE_CUDA_ARCHITECTURES})")
else()
  message(STATUS "fastllm: CPU-only build (no CUDA compiler found)")
endif()
# P4.4: WITHOUT this the aarch64 build defines neither __ARM_FEATURE_DOTPROD
# nor __ARM_FEATURE_MATMUL_INT8, so all 11 NEON int8 blocks in kernels.h (and
# their bitwise tests) compile out silently and the engine runs the SCALAR
# fallbacks. Verified on thor: `objdump` found 0 sdot/smmla in libfastllm.a.
# -ffp-contract=off above still governs the canonical path; the NEON kernels
# use explicit intrinsics, which these flags do not reassociate.
option(FASTLLM_ARCH_NATIVE "aarch64: enable dotprod/i8mm paths" ON)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND FASTLLM_ARCH_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=native" FASTLLM_HAS_MCPU_NATIVE)
  if(FASTLLM_HAS_MCPU_NATIVE)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mcpu=native>")
    message(STATUS "fastllm: aarch64 -mcpu=native (dotprod/i8mm paths live)")
  else()
    check_cxx_compiler_flag("-march=armv8.2-a+dotprod+i8mm" FASTLLM_HAS_V82)
    if(FASTLLM_HAS_V82)
      add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-march=armv8.2-a+dotprod+i8mm>")
      message(STATUS "fastllm: aarch64 armv8.2-a+dotprod+i8mm")
    endif()
  endif()
endif()

# Rebuild cost dominates phase turnaround on thor (nvcc, 14 cores): reuse a
# compiler cache when one is installed. Off automatically when absent.
find_program(FASTLLM_CCACHE ccache)
if(FASTLLM_CCACHE)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${FASTLLM_CCACHE})
  if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${FASTLLM_CCACHE})
  endif()
  message(STATUS "fastllm: ccache enabled (${FASTLLM_CCACHE})")
endif()
option(FASTLLM_TSAN "build with -fsanitize=thread" OFF)
if(FASTLLM_TSAN)
  add_compile_options(-fsanitize=thread -g -O1)
  add_link_options(-fsanitize=thread)
endif()
# Guarded: sibling components land on parallel branches; build what exists.
foreach(sub src/runtime src/bench tests tools)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${sub}/CMakeLists.txt)
        add_subdirectory(${sub})
    endif()
endforeach()
