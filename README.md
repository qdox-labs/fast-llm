# Fast-LLM — Work in progress

Fast-LLM is QDOX's work-in-progress C++17 research runtime for language-model inference. This source snapshot explores CPU/GPU execution, quantized weight access and speculative decoding. Its APIs and supported operations are evolving.

The source builds with GNU C++ 13.3 in Debug mode with CUDA disabled. The completed CPU checks are listed below. They do not establish performance, numerical equivalence on every platform, production readiness or robot safety.

## Included source

- A task graph, memory arena, CPU execution path and optional CUDA execution path.
- GGUF parsing, model/decoder code and quantized weight layouts and lookup tables.
- Draft-length selection and suffix-based draft proposals.
- Six test executables, a decoder command-line program and small generation/profiling utilities.

This snapshot contains no model weights, benchmark dataset, compiled library, runtime log or development-history archive. Model files must be supplied separately under their own terms. Support for one model family or quantization does not imply support for every GGUF file.

## Build prerequisites and checks

The build requires CMake 3.24 or newer, a C++17 compiler and a POSIX/Linux environment. CUDA support is enabled when CMake discovers a CUDA compiler. The current CUDA architecture default is 110; configure an architecture supported by the intended GPU and toolchain before building. The optional Python comparison utility requires NumPy and gguf-py supplied separately. No third-party runtime binaries are bundled.

The commands below describe the available build and test targets. CUDA discovery depends on the environment; the publication check explicitly disabled CUDA and used a separate build directory.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/tests/fastllm-tests
./build/tests/fastllm-quant-tests
./build/tests/fastllm-decode-tests
./build/tests/fastllm-spec-tests
./build/tests/fastllm-headcert-tests
./build/tests/fastllm-shadow-tests
```

Record the source revision, compiler, CMake configuration, device and each result before reporting a tested configuration. CUDA checks and performance measurements require their own reproducible runs. No speedup or measured throughput is claimed by this README.

The shadow-cache path can be set with `FASTLLM_SHADOW_FILE`. The default is `~/.fastllm-shadow.bin`; shadow mode is off unless explicitly enabled. A focused check verified default-path selection, empty and explicit overrides, and the default disabled mode without writing a cache file.

## Recorded CPU checks

A GNU C++ 13.3 Debug build with CUDA disabled completed. It emitted an existing aggregate-initialization warning in decoder code. Five completed test programs reported:

| Check group | Checks | Failures | Elapsed time |
|---|---:|---:|---:|
| Runtime | 158 | 0 | 1.65 s |
| Quantization | 39 | 0 | 0.02 s |
| Decoder operations | 27 | 0 | 0.001 s |
| Speculative helpers | 2,292 | 0 | 0.003 s |
| Shadow cache helpers | 198 | 0 | Not recorded |

The head-certificate program exceeded a 60-second bound and was stopped; it has no pass verdict. Source inspection shows billions of brute-force terms in its fixed test loops, so the timeout is compatible with workload size, but that observation is not a correctness result. These x86 CPU checks do not exercise ARM-specific paths or CUDA. No GPU or model-weight inference experiment was run for these publication checks.

## Research status

The next steps are completion of scoped validation, an explicit supported-model/operation matrix, and a reproducible evaluation with complete inputs and failure reporting. This repository is a research starting point; it is not a deployed robotics controller, agent service or memory product.

## License

Fast-LLM is released under the [MIT License](LICENSE), copyright 2026 Maneesh Disodia. The included ggml/llama.cpp-derived portions retain their upstream MIT notice; see [third-party notices](THIRD_PARTY_NOTICES.md). No model-weight or dataset license is implied.
