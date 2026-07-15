# Month 3 Week 2 Day 2 — Row-Range Batching

## Goal

Add contiguous row-range batching as the foundation for future multi-stream overlap.

## Batching Strategy

Each batch is a contiguous slice of rows.

Each batch computes partial GROUP BY count/sum results for all groups.

The CPU merges partial results after all batches finish.

## Why Row-Range Batching

Row-range batching avoids key partitioning, sorting, and skew-induced load imbalance. It keeps this step focused on pipeline scheduling rather than semantic partitioning.

## What Changed

- Added `--batch-rows`
- Added sequential batch processing
- Added partial result storage
- Added CPU merge of partial count/sum arrays
- Preserved sync and single-stream async execution modes
- Device input buffers sized to `max_batch_rows`
- Phase timings accumulate across all batches in an iteration
- Added `avg_cpu_merge_ms` reporting
- Added batch-level NVTX ranges (`batched_pipeline`, `batch_N`, `cpu_merge_partial_results`)

## What Did Not Change

- No multi-stream overlap
- No double buffering
- No key-based partitioning
- No kernel algorithm change
- No data layout change

## Verification

Build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

### Non-batched baseline (pinned, single-stream-async)

```text
batch_rows:   0
num_batches:  1
avg_total_gpu_ms: 31.6358
avg_h2d_ms: 13.7829
avg_block_partial_reduce_kernel_ms: 15.2301
avg_cpu_merge_ms: 0.0029
```

### Batched sync (`--batch-rows 3000000`)

```text
batch_rows:   3000000
num_batches:  4
avg_total_gpu_ms: 31.6972
avg_h2d_ms: 11.9644
avg_block_partial_reduce_kernel_ms: 16.7728
avg_cpu_merge_ms: 0.0083
```

### Batched single-stream async (`--batch-rows 3000000`)

```text
batch_rows:   3000000
num_batches:  4
avg_total_gpu_ms: 31.5059
avg_h2d_ms: 11.7852
avg_block_partial_reduce_kernel_ms: 16.7834
avg_cpu_merge_ms: 0.0069
```

### Validation

Batched block-partial and atomic modes with `--validate true` both report `all_keys_correct: 1`.

Pageable sync batched validation also passes.

### Nsight Systems capture

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --output=/tmp/week2_day2_batched_sequential \
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
    --execution single-stream-async \
    --batch-rows 3000000
```

Observed counts (warmup + 3 measured = 4 pipeline executions × 4 batches = 16):

- Kernel instances: 16 per pipeline kernel
- H→D memcpys: 48 (16 batches × 3 columns)
- D→H memcpys: 48 (16 batches × 3 transfers)
- `cudaMemcpyAsync`: 96
- NVTX: `batch_0`/`batch_1`/`batch_2`/`batch_3` each 4 instances; `batched_pipeline` 4; `cpu_merge_partial_results` 4

Profiled timing:

```text
avg_total_gpu_ms: 30.8162
avg_h2d_ms: 11.8029
avg_block_partial_reduce_kernel_ms: 15.7527
avg_cpu_merge_ms: 0.0078
```

## Interpretation

The pipeline can now process contiguous row ranges as sequential batches. Each batch produces partial GROUP BY count/sum results for all groups, and the CPU merges partial results into the final output. This validates the batching abstraction needed for future multi-stream overlap, but execution is still serialized.

## Status

Verified on 2026-07-15.
