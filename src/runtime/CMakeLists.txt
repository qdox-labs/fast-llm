find_package(Threads REQUIRED)
add_library(fastllm STATIC arena.cpp graph.cpp runtime.cpp shadow.cpp
            ${CMAKE_SOURCE_DIR}/src/model/gguf.cpp
            ${CMAKE_SOURCE_DIR}/src/model/decoder.cpp)
if(CMAKE_CUDA_COMPILER)
  target_sources(fastllm PRIVATE gpu_exec.cu)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(fastllm PUBLIC CUDA::cudart)
endif()
target_include_directories(fastllm
  PUBLIC  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fastllm PUBLIC Threads::Threads)
target_compile_options(fastllm PRIVATE -O2 -Wall -Wextra)
