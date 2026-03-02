# KTransformers GPU Fallback Buffer Release

## Problem

When using `--kt-gpu-prefill-token-threshold` to enable GPU fallback for MoE
expert computation during prefill, KTransformers allocates a temporary GPU
buffer to hold one full layer's expert weights (all 512 experts for FP8
models like Qwen3.5-397B-A17B). This buffer consumes approximately **6.3 GB
of VRAM** for a typical configuration.

By default, this buffer is allocated on the first GPU fallback and **never
released**, permanently reducing the VRAM available for other operations. On
memory-constrained setups (e.g., a single 32 GB GPU), this can cause
out-of-memory errors during:

- Vision-Language (VL) processing, where the ViT encoder needs activation
  memory.
- Linear attention (Gated DeltaNet) computation within the same forward pass,
  which requires its own activation memory.

Two command-line flags provide control over when this buffer is released.

## `--kt-gpu-fallback-release`

Releases the GPU expert weight buffer after the **last MoE layer** completes
its GPU fallback computation in each forward pass.

**When to use:** You are running a Vision-Language model and encounter OOM
errors during ViT processing after a GPU fallback prefill pass. The ViT runs
after the language model's forward pass completes, so releasing the buffer
after the last layer gives the ViT ~6 GB of additional headroom.

**Overhead:** Minimal. On the next GPU fallback, the buffer is re-allocated
via `torch.empty()` (~1-5 ms) before expert weights are loaded. This is
negligible compared to the ~250 ms per-layer weight transfer time.

**Example:**

```bash
python -m sglang.launch_server \
    --model ./models/Qwen3.5-397B-A17B-FP8 \
    --kt-gpu-prefill-token-threshold 2048 \
    --kt-gpu-fallback-release \
    ...
```

## `--kt-gpu-fallback-per-layer-release`

Releases the GPU expert weight buffer after **every layer's** MoE compute
during GPU fallback, not just after the last layer. The buffer is re-allocated
before the next layer's MoE operation.

This means that between MoE operations — when attention kernels (full
attention or linear attention like Gated DeltaNet) execute — the ~6.3 GB
buffer is not present, giving attention layers significantly more VRAM for
activations.

Implies `--kt-gpu-fallback-release` (the last-layer flush via
`torch.cuda.empty_cache()` always runs).

**When to use:** You encounter OOM errors during prefill in attention layers
(e.g., `chunk_gated_delta_rule` or similar) that need activation memory while
the expert buffer is allocated. This is common with large
`--chunked-prefill-size` values (e.g., 16384+ tokens). Enabling this flag
allows you to increase chunk size without OOM.

**Overhead:** Each layer incurs a buffer release (~1 ms) and re-allocation
(~1-5 ms). Across 60 layers, this adds ~60-300 ms per forward pass. The
expensive `torch.cuda.empty_cache()` is only called once (after the last
layer), not per-layer. The per-layer overhead is negligible compared to the
~250 ms per-layer weight transfer.

**Example:**

```bash
python -m sglang.launch_server \
    --model ./models/Qwen3.5-397B-A17B-FP8 \
    --kt-gpu-prefill-token-threshold 2048 \
    --kt-gpu-fallback-per-layer-release \
    --chunked-prefill-size 32768 \
    ...
```

## Flag Interaction

| Flags | Behavior |
|---|---|
| Neither | Original behavior: buffer allocated once, never released |
| `--kt-gpu-fallback-release` | Buffer released after last layer; ~6 GB freed for post-forward ops (ViT) |
| `--kt-gpu-fallback-per-layer-release` | Buffer released after every layer; ~6 GB freed between MoE and attention ops |

`--kt-gpu-fallback-per-layer-release` implies `--kt-gpu-fallback-release`.
If both are passed, the per-layer behavior takes effect.

## Technical Details

The implementation lives in `SharedFullContext` within
`sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py`:

- `release_gpu_weights(flush_cache)`: Replaces GPU weight tensor data with
  zero-size tensors. Records tensor metadata (shape, dtype, device) so
  re-allocation can restore the correct sizes. When `flush_cache=True`,
  calls `torch.cuda.empty_cache()` to return blocks to the CUDA driver.
- `_ensure_gpu_weights_allocated()`: Checks `_weights_released` and, if
  needed, re-allocates weight tensors from saved metadata before the next
  `load()` call fills them with expert data.

The CPU-side resources (POSIX shared memory buffers, pinned memory
registrations, cross-rank buffer pointers) are never released, avoiding the
expensive re-initialization cost.
