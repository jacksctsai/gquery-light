# Month 3 Week 2 Day 3 — Multi-Stream Batched Execution

## Goal

Introduce the first multi-stream batched execution mode for G-Query Light.

## What Changed

- Added `--execution multi-stream-batched`
- Added `--num-streams`
- Added one CUDA stream context per stream
- Added per-stream device buffers
- Assigned row batches to streams (`stream_id = batch_id % num_streams`)
- Preserved CPU merge of partial count/sum results
- Multi-stream D→H partial results use pinned host slices
- Total multi-stream elapsed time uses a CPU timer (`avg_total_gpu_ms`); per-phase CUDA event fields remain zero in this mode

## Execution Model

Each batch is a contiguous row range.

Each batch computes partial count/sum results for all groups.

Batches are assigned to streams by:

```text
stream_id = batch_id % num_streams
```

Each stream owns independent device buffers.

Safe reuse policy: synchronize a stream before enqueuing a later batch onto the same stream context.

## What Did Not Change

- No key partitioning
- No sorting
- No GROUP BY semantic change
- No GPU final merge
- No kernel algorithm change
- No data layout change
- Existing `sync` and `single-stream-async` paths preserved
- Sequential batching preserved

## Important Note

Multi-stream structure does not automatically prove overlap.

Overlap must be verified with Nsight Systems.

## Verification

Build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

### CLI validation

```text
error: multi-stream-batched requires --memory pinned because async H→D/D→H transfer overlap requires pinned host memory.
error: multi-stream-batched requires --batch-rows > 0.
error: multi-stream-batched requires --num-streams >= 2
```

### Validation (block-partial and atomic)

Both modes with `--validate true` report `all_keys_correct: 1`.

### Multi-stream batched run

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
  --execution multi-stream-batched \
  --batch-rows 3000000 \
  --num-streams 2
```

Observed config / timing:

```text
execution:    multi-stream-batched
batch_rows:   3000000
num_batches:  4
num_streams:  2
avg_total_gpu_ms: 22.1880
avg_cpu_merge_ms: 0.0037
```

Existing single-stream async sequential batching for reference:

```text
avg_total_gpu_ms: 31.4619
avg_cpu_merge_ms: 0.0079
```

Wall-clock total improved in this run, but overlap is not claimed from timing alone.

### Nsight Systems capture

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --output=/tmp/week2_day3_multi_stream_batched \
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
    --execution multi-stream-batched \
    --batch-rows 3000000 \
    --num-streams 2
```

Observed counts (warmup + 3 measured = 4 pipeline executions × 4 batches = 16):

- Kernel instances: 16 per pipeline kernel
- H→D memcpys: 48 (16 batches × 3 columns)
- D→H memcpys: 48 (16 batches × 3 transfers)
- `cudaMemcpyAsync`: 96
- `cudaStreamSynchronize`: 32
- NVTX: `batch_0_stream_0` / `batch_1_stream_1` / `batch_2_stream_0` / `batch_3_stream_1` each 4 instances
- NVTX: `multi_stream_batched_pipeline` 4; `cpu_merge_partial_results` 4; `execution_multi_stream_batched` 2

Profiled timing:

```text
avg_total_gpu_ms: 22.3755
avg_cpu_merge_ms: 0.0043
```

## Interpretation

The pipeline now supports multi-stream batched execution. Each row batch is assigned to a CUDA stream, each stream owns independent device buffers, and partial count/sum outputs are copied into unique host slices before CPU merge. This creates the first execution structure where H→D transfer for one batch may overlap with compute for another batch. Actual overlap must be verified in Nsight Systems timeline and is not assumed from code structure alone.

## Status

Verified on 2026-07-15.
