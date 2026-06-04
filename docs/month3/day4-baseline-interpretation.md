## Pre-Day-4 Baseline Interpretation

### Where is the GPU idle?

The GPU compute engines are idle before the first GPU operation while the CPU synthesizes input data.

During H→D and D→H transfers, the SMs are idle because no kernel is executing, but the copy engine may be active. Therefore these regions should be described as compute-engine idle time, not full-device idle time.

Any visible gaps between H→D, kernels, and D→H should be treated as possible orchestration or synchronization idle time until confirmed in the Nsight Systems timeline.

### Are H→D, kernels, and D→H serialized?

Yes. The current baseline executes H→D transfer, kernels, and D→H transfer sequentially. There is no intended overlap yet because the implementation does not use CUDA streams, async copies, pinned host memory, or staged multi-batch scheduling.

### Which phase dominates at 1M, 12M, and 64M?

H→D transfer is the dominant phase in most measured cases.

At 1M uniform atomic, total GPU pipeline time is about 1.98 ms and H→D is about 1.84 ms.

At 12M hotkey block-partial, total GPU pipeline time is about 22.94 ms. H→D is about 12.30 ms and block-partial reduction is about 8.06 ms. This is a mixed case: H→D is still the largest phase, but reduction is also significant.

At 64M uniform block-partial, total GPU pipeline time is about 83.38 ms and H→D is about 67.50 ms, so transfer strongly dominates.

### Does hotkey shift the bottleneck from transfer toward reduction?

Not proven yet.

H→D transfer should be mostly insensitive to key distribution because the same number of rows and bytes are transferred.

Kernel time may change with distribution. In atomic mode, hotkey can increase contention because many rows update the same group. In block-partial mode, the effect depends on the implementation: local aggregation, scan/scatter behavior, memory access pattern, and partial reduction structure.

To prove a distribution-driven bottleneck shift, we need same-size comparisons across uniform, hotkey, and zipf using the same reduction mode.

### What can and cannot be concluded before streams?

We can conclude that the baseline is serialized and transfer-heavy. We can also conclude that overlapping transfer and compute is a reasonable Week 2 hypothesis.

We cannot conclude that streams alone will improve performance. Useful overlap requires independent batches, async copies, pinned host memory, compatible hardware copy/compute concurrency, and removal of synchronization barriers.

We also cannot predict speedup yet. The speedup depends on how much transfer time can actually be hidden behind compute and whether batching introduces new overhead.