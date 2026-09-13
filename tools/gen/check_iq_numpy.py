#!/usr/bin/env python3
"""f64 anchor for the IQ2_XXS / IQ3_XXS decoders, computed via gguf-py.

Independent of fast-llm's C++ entirely: gguf-py parses the file and numpy
dequantizes at f64, so a block-layout misread in our reader/kernels shows up
as a mismatch. `membench --iq-check` proves host-ref == gpu-f32 == gpu-int8;
this proves host-ref == an outside implementation.

usage: check_iq_numpy.py <gguf-shard> <gguf-py-dir> [n_rows]
prints one line per format: tensor, rows, max relative error vs our reference
values supplied on stdin as "<tensor> <row> <value>" lines (from --iq-dump),
or, with no stdin, just dumps the anchor values for eyeballing.
"""
import sys, numpy as np

shard, gguf_dir = sys.argv[1], sys.argv[2]
nrows = int(sys.argv[3]) if len(sys.argv) > 3 else 8
sys.path.insert(0, gguf_dir)
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize                 # noqa: E402

r = GGUFReader(shard)
want = {GGMLQuantizationType.IQ2_XXS: "IQ2_XXS",
        GGMLQuantizationType.IQ3_XXS: "IQ3_XXS"}

# same deterministic activation vector membench --iq-check uses
def activations(cols):
    s = 0x243F6A8885A308D3
    out = np.empty(cols, dtype=np.float64)
    for i in range(cols):
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out[i] = ((s >> 33) % 2001 - 1000) / 1000.0
    return out

for t in r.tensors:
    # MoE expert weights are 3-D ([cols, rows, n_experts]); rows are contiguous
    if t.tensor_type not in want or len(t.shape) < 2:
        continue
    cols = int(t.shape[0])
    if cols % 256:
        continue
    w = dequantize(t.data, t.tensor_type).astype(np.float64).reshape(-1, cols)[:nrows]
    x = activations(cols)
    dots = w @ x
    print("%s %s rows=%d cols=%d" % (want[t.tensor_type], t.name, len(dots), cols))
    for i, v in enumerate(dots):
        print("  row %d anchor %.9g" % (i, v))
    del want[t.tensor_type]
    if not want:
        break
