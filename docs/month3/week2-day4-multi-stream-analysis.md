# Month 3 Week 2 Day 4 — Multi-Stream Batching Analysis

## Goal

Analyze whether multi-stream row-range batching reduced exposed GPU pipeline time compared with sequential row-range batching.

This day is analysis-only. No new kernel algorithm, data layout, or batching strategy is introduced.

---

## Compared Configurations

Both runs used the same workload:

| Field | Value |
|---|---:|
| `num_rows` | 12,000,000 |
| `num_groups` | 1024 |
| `distribution` | hotkey |
| `mode` | block-partial |
| `memory` | pinned |
| `batch_rows` | 3,000,000 |
| `num_batches` | 4 |
| `warmup` | 1 |
| `iterations` | 3 |
| `validate` | false |

The main difference was the execution mode.

| Version | Execution Mode | Streams |
|---|---|---:|
| Sequential batched | `single-stream-async` | 1 |
| Multi-stream batched | `multi-stream-batched` | 2 |

---

## Timing Comparison

| Version | Execution | Streams | `avg_total_gpu_ms` |
|---|---|---:|---:|
| Week 2 Day 2 | `single-stream-async` | 1 | ~31.86 ms |
| Week 2 Day 3 | `multi-stream-batched` | 2 | ~22.38 ms |

The multi-stream batched path reduced exposed total GPU time by approximately:

```text
31.86 ms - 22.38 ms = 9.48 ms
```

Relative reduction:

```text
9.48 / 31.86 ≈ 29.8%
```

This is a meaningful reduction for the same workload.

---

## Day 3 Program Output

The Day 3 multi-stream batched run reported:

```text
execution:    multi-stream-batched
batch_rows:   3000000
num_batches:  4
num_streams:  2

avg_total_gpu_ms: 22.3835
avg_h2d_ms: 0.0000
avg_filter_kernel_ms: 0.0000
avg_scan_kernel_ms: 0.0000
avg_scatter_kernel_ms: 0.0000
avg_block_partial_reduce_kernel_ms: 0.0000
avg_d2h_ms: 0.0000
avg_cpu_merge_ms: 0.0040
```

The CPU merge cost remained negligible.

---

## Nsight Structural Verification

Expected batch executions:

```text
(1 warmup + 3 measured iterations) × 4 batches = 16 batch executions
```

Nsight confirmed the expected structure.

Observed kernel count:

```text
groupby_selected_block_partial_kernel: 16 instances
```

Observed memory operation counts:

```text
H→D memcpy: 48 copies
D→H memcpy: 48 copies
CUDA memset: 48 operations
```

Observed stream-specific NVTX labels:

```text
batch_0_stream_0
batch_1_stream_1
batch_2_stream_0
batch_3_stream_1
```

This confirms that batches were assigned across two CUDA streams as intended.

---

## Interpretation

The result shows that multi-stream batching reduced exposed end-to-end GPU pipeline time for the tested workload.

The reduction from approximately `31.86 ms` to `22.38 ms` is consistent with the hypothesis that issuing independent row batches across multiple streams can reduce visible serialization between transfer and compute stages.

However, Nsight summary tables alone do not prove exact transfer/compute overlap.

To prove exact overlap, the Nsight Systems timeline must be inspected visually to check whether H→D memcpy intervals overlap with kernel execution from another stream.

---

## What We Can Conclude

- Row-range batching works.
- CPU merge cost is negligible.
- Multi-stream batched execution is structurally active.
- The multi-stream path reduced measured total GPU time compared with sequential batching.
- Independent per-stream buffers make concurrent stream scheduling safe.
- Batch partial outputs are copied into independent host slices before CPU merge.
- Nsight confirms the expected kernel and memory operation counts.

---

## What We Cannot Conclude Yet

- We cannot claim full H→D / compute overlap.
- We cannot claim zero-idle GPU.
- We cannot claim the implementation is optimal.
- We cannot claim `num_streams = 2` is the best stream count.
- We cannot claim `batch_rows = 3,000,000` is the best batch size.
- We cannot rely on per-phase program timing in multi-stream mode because phase fields currently report `0.0000`.

---

## Timing Note

In `multi-stream-batched` mode, per-phase CUDA event timing is currently not accumulated and reports zero:

```text
avg_h2d_ms: 0.0000
avg_filter_kernel_ms: 0.0000
avg_scan_kernel_ms: 0.0000
avg_scatter_kernel_ms: 0.0000
avg_block_partial_reduce_kernel_ms: 0.0000
avg_d2h_ms: 0.0000
```

For multi-stream mode, Nsight Systems should be treated as the source of truth for phase breakdown and overlap analysis.

The program-level `avg_total_gpu_ms` remains useful as an exposed end-to-end GPU pipeline time measurement.

---

## Current Bottleneck View

Before multi-stream batching, the serialized batched pipeline exposed transfer and compute stages more directly.

After multi-stream batching, total exposed time decreased, suggesting that some portion of previously exposed serialized time was hidden or scheduled more efficiently.

The dominant kernel remains:

```text
groupby_selected_block_partial_kernel
```

The dominant GPU memory operation remains:

```text
CUDA memcpy Host-to-Device
```

The next question is not whether the mechanism works. It does.

The next question is how sensitive the result is to batch size and stream count.

---

## Next Experiment

Run a small sweep while keeping the workload fixed:

```text
num_rows: 12,000,000
distribution: hotkey
mode: block-partial
memory: pinned
```

Suggested sweep:

| `batch_rows` | `num_streams` |
|---:|---:|
| 1,500,000 | 2 |
| 3,000,000 | 2 |
| 6,000,000 | 2 |
| 3,000,000 | 4 |

The goal is to characterize whether performance is sensitive to batch size and stream count.

---

## Summary

Sequential row-range batching validated correctness.

Multi-stream row-range batching reduced exposed total GPU pipeline time from approximately `31.86 ms` to `22.38 ms`.

This is the first meaningful evidence that the pipeline can benefit from multi-stream scheduling.

The pipeline now supports multi-stream row-range batching, and the measured exposed GPU pipeline time decreased meaningfully compared with sequential batching. Exact transfer/compute overlap still requires Nsight Systems timeline confirmation.

