# Third-party notices

## ggml / llama.cpp

Parts of the quantized block layouts, scalar reconstruction helpers, IQ lookup tables and GGUF type conventions derive from ggml/llama.cpp. The model decoder also identifies llama.cpp's DeepSeek2 graph and YaRN conventions as its architectural reference. The upstream MIT terms are retained verbatim in [licenses/ggml-MIT.txt](licenses/ggml-MIT.txt).

Public comparison snapshot: [ggml-org/llama.cpp at 5f436dddb440a288ee5611d7d1eca564a6aca9f4](https://github.com/ggml-org/llama.cpp/tree/5f436dddb440a288ee5611d7d1eca564a6aca9f4). Source review matched six format layouts and scalar reconstruction formulas, the Q4 scale/min unpacker and eight GGUF type identifiers. The four lookup-table sequences match all 648 integer entries in that public snapshot. The local implementation combines reconstructed values with task-graph and dot-product operations; this attribution does not claim byte-identical implementation or numerical parity on every target.

The exact historical adaptation revision is not recorded. The linked revision identifies the public source used for this provenance review, not an invented import date or original commit.

Relevant upstream sources: [format layouts and tables](https://github.com/ggml-org/llama.cpp/blob/5f436dddb440a288ee5611d7d1eca564a6aca9f4/ggml/src/ggml-common.h), [scalar quantization routines](https://github.com/ggml-org/llama.cpp/blob/5f436dddb440a288ee5611d7d1eca564a6aca9f4/ggml/src/ggml-quants.c), [GGML type identifiers](https://github.com/ggml-org/llama.cpp/blob/5f436dddb440a288ee5611d7d1eca564a6aca9f4/ggml/include/ggml.h), [DeepSeek2 model graph](https://github.com/ggml-org/llama.cpp/blob/5f436dddb440a288ee5611d7d1eca564a6aca9f4/src/models/deepseek2.cpp), [shared graph construction](https://github.com/ggml-org/llama.cpp/blob/5f436dddb440a288ee5611d7d1eca564a6aca9f4/src/llama-graph.cpp), [CPU operation helpers](https://github.com/ggml-org/llama.cpp/blob/5f436dddb440a288ee5611d7d1eca564a6aca9f4/ggml/src/ggml-cpu/ops.cpp), and [MIT license](https://github.com/ggml-org/llama.cpp/blob/5f436dddb440a288ee5611d7d1eca564a6aca9f4/LICENSE).

## External tools and libraries

The standard C++ library, POSIX threads, CMake and compiler are supplied by the build environment. The optional CUDA build uses the NVIDIA CUDA Toolkit and runtime; neither is redistributed in this source candidate. [NVIDIA's SDK terms](https://docs.nvidia.com/cuda/eula/index.html) govern separately supplied NVIDIA components.

The optional Python comparison utility imports NumPy and gguf-py; they are not bundled. NumPy publishes its [BSD-style license](https://numpy.org/doc/stable/license.html). gguf-py belongs to the upstream llama.cpp project. Pin and record the actual tool versions when publishing reproducibility instructions.

Model weights and data are not included. Their publishers' licenses and usage conditions must be assessed separately for any future example or dataset release.
