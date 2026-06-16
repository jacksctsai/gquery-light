# Month 3 Week 2 Day 1 — Single-Stream Async Baseline

## Goal

Add an async-capable execution path without changing the logical serialized pipeline.

## What Changed

- Added `--memory pageable|pinned`
- Added `--execution sync|single-stream-async`
- Added pinned host allocation path (`HostVector` with `cudaMallocHost`)
- Added explicit CUDA stream path (`CudaStreamGuard`)
- Added `cudaMemcpyAsync` path (single-stream-async mode only)
- Launched kernels into the explicit stream in async mode
- Used stream-level synchronization (`cudaStreamSynchronize`)

## What Did Not Change

- No batching
- No double buffering
- No multi-stream overlap
- No kernel algorithm changes
- No data layout changes

## Expected Behavior

The single-stream async path remains serialized:

```text
H→D → kernels → D→H → stream sync
```

This is intentional. Because H2D, kernels, and D2H are issued into the same stream, they preserve order and are not expected to overlap. Multi-stream batching will be introduced in a later step.

## Verification

Build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

### Sync baseline (pageable, block-partial, 12M rows)

```bash
./build/gquery \
  --num-rows 12000000 \
  --num-groups 1024 \
  --distribution hotkey \
  --mode block-partial \
  --warmup 1 \
  --iterations 3 \
  --seed 42 \
  --validate false \
  --memory pageable \
  --execution sync
```

Sample output:

```text
memory:       pageable
execution:    sync
avg_total_gpu_ms: 72.5215
avg_h2d_ms: 61.9705
avg_filter_kernel_ms: 0.4058
avg_scan_kernel_ms: 1.3769
avg_scatter_kernel_ms: 0.8029
avg_block_partial_reduce_kernel_ms: 7.7721
avg_d2h_ms: 0.1776
```

### Pinned single-stream async (block-partial, 12M rows)

```bash
./build/gquery \
  --num-rows 12000000 \
  --num-groups 1024 \
  --distribution hotkey \
  --mode block-partial \
  --warmup 1 \
  --iterations 3 \
  --seed 42 \
  --validate false \
  --memory pinned \
  --execution single-stream-async
```

Sample output:

```text
memory:       pinned
execution:    single-stream-async
avg_total_gpu_ms: 23.4947
avg_h2d_ms: 13.1075
avg_filter_kernel_ms: 0.4077
avg_scan_kernel_ms: 1.3089
avg_scatter_kernel_ms: 0.8062
avg_block_partial_reduce_kernel_ms: 7.8129
avg_d2h_ms: 0.0359
```

Pinned async shows faster H2D/D2H timing vs pageable sync, but the pipeline remains serialized on one stream.

### Atomic mode with validation

```bash
./build/gquery \
  --num-rows 1000000 \
  --distribution uniform \
  --mode atomic \
  --warmup 1 \
  --iterations 3 \
  --validate true \
  --memory pinned \
  --execution single-stream-async
```

Validation passed (all per-key count/sum deltas zero).

### Pageable + single-stream-async warning

```bash
./build/gquery --num-rows 1000 --mode atomic --warmup 0 --iterations 1 \
  --validate false --memory pageable --execution single-stream-async
```

Prints:

```text
Warning: single-stream-async with pageable memory may not provide true async host transfer behavior. Use --memory pinned for async transfer experiments.
```

### Nsight Systems capture

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --output=/tmp/week2_day1_single_stream_async \
  ./build/gquery \
    --num-rows 12000000 \
    --num-groups 1024 \
    --distribution hotkey \
    --mode block-partial \
    --warmup 1 \
    --iterations 3 \
    --seed 42 \
    --validate false \
    --memory pinned \
    --execution single-stream-async
```

`cudaMemcpyAsync` appears in `cuda_api_sum`; NVTX ranges include `execution_single_stream_async`, `host_memory_pinned`, `h2d_transfer`, `filter_kernel_launch`, `block_partial_reduce_kernel_launch`, `d2h_transfer`.

## Status

Verified on 2026-06-16. Build succeeds, defaults remain `--memory pageable --execution sync`, validation passes, CUDA event phase timing prints in all modes, and Nsight Systems capture confirms async memcpy and NVTX ranges.
