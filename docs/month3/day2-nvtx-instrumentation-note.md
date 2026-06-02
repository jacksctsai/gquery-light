# Month 3 Day 2 — NVTX Instrumentation Result

## Status

Day 2 acceptance criteria passed.

## Summary

NVTX instrumentation was added without changing benchmark behavior. Nsight Systems captures now show readable ranges for CPU data generation, warmup iterations, measured iterations, H→D transfer, kernel launch regions, D→H transfer, validation, allocation/free, and timing summary.

## Key Finding

CPU-side synthetic data generation dominates full program wall time.

For the 12M hotkey block-partial capture:

- `program_total`: ~2.37 s
- `cpu_generate_input`: ~2.01 s
- `avg_total_gpu_ms`: ~21.45 ms

This confirms that CPU data generation and GPU pipeline execution must be analyzed separately.

## Timeline Interpretation

The NVTX kernel launch ranges mark CPU launch activity, not actual GPU execution duration. Actual kernel execution must be interpreted from the CUDA GPU rows in Nsight Systems.

## Validation Behavior

`cpu_validate` appears only when validation is enabled. This confirms profiling-friendly runs can exclude validation from the measured path.

## Next Step

Proceed to Day 3: add CUDA event timing for H→D transfer, kernel execution phases, reduction stages, D→H transfer, and total GPU pipeline duration.
