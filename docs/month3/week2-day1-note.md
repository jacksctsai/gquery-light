## Week 2 Day 1 Verification Result

Three execution configurations were tested on the same 12M hotkey block-partial workload.

| Memory | Execution | avg_total_gpu_ms | avg_h2d_ms | avg_block_partial_reduce_kernel_ms |
|---|---|---:|---:|---:|
| pageable | sync | 44.3852 | 32.2556 | 9.5644 |
| pinned | sync | 33.1029 | 14.2756 | 15.2391 |
| pinned | single-stream-async | 32.5822 | 14.3313 | 15.3257 |

Pinned host memory reduced H→D transfer time substantially.

Single-stream async execution produced timing similar to pinned sync execution, which is expected because all operations are issued into one stream and remain serialized.

This confirms that the codebase is now async-capable, but not yet overlapping transfer and compute.

## Week 2 Day 1 — Nsight Verification

The `single-stream-async` path was captured with Nsight Systems.

The CUDA API summary confirms that the async path is active:

- `cudaMemcpyAsync`: 24 calls
- `cudaStreamSynchronize`: 8 calls
- `cudaStreamCreate`: 2 calls
- `cudaStreamDestroy`: 2 calls

The GPU kernel summary shows 4 instances of each pipeline kernel, matching 1 warmup plus 3 measured iterations.

The GPU memory summary shows 12 H→D copies and 12 D→H copies.

Measured program timing:

- `avg_total_gpu_ms`: 31.8582 ms
- `avg_h2d_ms`: 14.3305 ms
- `avg_block_partial_reduce_kernel_ms`: 14.7591 ms
- `avg_d2h_ms`: 0.0581 ms

Interpretation:

The pipeline is now async-capable, using pinned memory, `cudaMemcpyAsync`, and an explicit CUDA stream. However, because all operations are issued into one stream, execution remains serialized. This step prepares the codebase for multi-stream batching but does not implement overlap yet.

Note:

NVTX range durations around async calls reflect CPU enqueue time, not GPU transfer duration. For async transfer duration, use CUDA event timing and GPU MemOps summaries.