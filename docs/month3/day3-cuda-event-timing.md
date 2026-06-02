# Month 3 Day 3 — CUDA Event Timing Result

## Status

Day 3 acceptance criteria passed.

## What Changed

The benchmark now reports CUDA event phase timings for measured iterations only.

The output includes:

- `avg_total_gpu_ms`
- `avg_h2d_ms`
- mode-specific kernel timing
- `avg_d2h_ms`

Atomic mode reports:

- `avg_atomic_groupby_kernel_ms`

Block-partial mode reports:

- `avg_filter_kernel_ms`
- `avg_scan_kernel_ms`
- `avg_scatter_kernel_ms`
- `avg_block_partial_reduce_kernel_ms`

`final_reduce_kernel_ms` is currently unused on the active group-by path because there is no separate final-reduce kernel in that path.

## Measurement Scope

CUDA event timing includes:

- H→D transfer
- GPU kernels
- D→H transfer
- total serialized GPU pipeline

CUDA event timing excludes:

- CPU synthetic data generation
- CPU validation
- printing / timing summary

Warmup iterations are excluded from timing averages.

## Sample Results

| Run | avg_total_gpu_ms | Dominant phase |
|---|---:|---|
| 1M uniform atomic | ~1.98 | H→D transfer, ~1.84 ms |
| 12M hotkey block-partial | ~22.94 | Mixed: H→D ~12.30 ms, block partial reduce ~8.06 ms |
| 64M uniform block-partial | ~83.38 | H→D transfer, ~67.50 ms |

## Initial Interpretation

The serialized baseline is transfer-heavy, especially for larger inputs.

For the 64M uniform block-partial case, H→D accounts for roughly 81% of measured GPU pipeline time.

For the 12M hotkey block-partial case, H→D remains the largest phase, but the block-partial reduction kernel is also significant.

This suggests Week 2 overlap work should focus on hiding H→D transfer behind compute where possible, instead of only optimizing kernels.

## Verification

The following checks passed:

- Build succeeds.
- Smoke test passes.
- Day 1 CLI behavior is preserved.
- Day 2 NVTX behavior is preserved.
- Warmup is excluded from phase timing.
- No streams were introduced.
- No async memcpy was introduced.
- No pinned memory was introduced.
- No data layout or kernel algorithm changes were introduced.