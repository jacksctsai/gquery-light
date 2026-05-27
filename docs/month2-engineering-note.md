# Month 2 Engineering Note — GPU Filtering, Compaction, Reduction, and GROUP BY

## Overview

Month 2 focused on moving G-Query Light from a simple GPU execution prototype toward a more realistic GPU-native query pipeline.

The goal was not only to make the GPU version faster, but to understand **why performance changes** as dataset size, aggregation strategy, and key distribution change.

The work covered:

- GPU filtering
- device-side mask generation
- prefix-scan based compaction
- selected-row reduction
- atomic vs block-partial aggregation
- GROUP BY over `passenger_count`
- key distribution stress tests

The main technical lesson was:

> GPU performance is often limited less by raw computation and more by memory movement, synchronization structure, and contention patterns.

---

## 1. Data Layout and Query Shape

The system uses a Structure-of-Arrays layout:

```text
trip_distance[]
fare_amount[]
passenger_count[]
```

This layout allows adjacent GPU threads to read adjacent memory addresses, improving memory coalescing during scan-style kernels.

The main predicate used throughout Month 2 was:

```sql
trip_distance > 2.5 AND fare_amount > 10.0
```

The later GROUP BY query shape was:

```sql
SELECT passenger_count, SUM(fare_amount)
FROM trips
WHERE trip_distance > 2.5
  AND fare_amount > 10.0
GROUP BY passenger_count;
```

---

## 2. Initial Atomic Filter Baseline

The initial GPU baseline used one thread per row.

For each matching row, the kernel performed global atomic updates:

```cpp
atomicAdd(count, 1ULL);
atomicAdd(sum_fare, fare_amount[i]);
```

This implementation was useful because it was simple and easy to validate against the CPU reference.

However, it mixed two different concerns:

1. predicate filtering
2. aggregation

That made it difficult to understand whether runtime was dominated by filtering work or by global atomic contention.

### Baseline Characteristics

Advantages:

- simple implementation
- minimal pipeline complexity
- easy correctness checking

Disadvantages:

- all matching rows contend on the same global memory locations
- runtime grows with selected-row count
- launch configuration changes do not help much once contention dominates

---

## 3. Mask-Only Filtering

To isolate the cost of filtering, the pipeline was refactored to generate a device-side mask:

```cpp
mask[i] = predicate ? 1 : 0;
```

This removed global aggregation from the filter kernel.

The mask-only kernel showed that predicate evaluation itself was very fast.

For tens of millions of rows, filter time dropped from tens of milliseconds in the atomic baseline to roughly 1–3 ms in the mask-only path.

This demonstrated that the original bottleneck was not the predicate scan. The real bottleneck was contention from global atomic aggregation.

### Key Observation

> Removing global atomics turned filtering into a regular memory-throughput kernel.

---

## 4. Filter Scaling Results

A dataset-size sweep showed the difference between atomic filtering and mask-only filtering.

| Dataset | Selectivity | Atomic Filter ms | Mask Filter ms | Ratio |
|---:|---:|---:|---:|---:|
| 1M | ~0.354 | ~1.32 | ~0.06 | ~20x |
| 5M | ~0.338 | ~5.65 | ~0.21 | ~27x |
| 10M | ~0.342 | ~12.19 | ~0.37 | ~33x |
| 25M | ~0.351 | ~21.98 | ~0.88 | ~25x |
| 50M | ~0.344 | ~35.20 | ~1.72 | ~21x |

The mask-only path scaled much more cleanly.

The atomic path remained much slower because roughly one third of rows matched the predicate, causing millions of global atomic operations.

### Interpretation

The atomic implementation had many parallel threads, but those threads were not making independent progress. They were converging onto shared global counters.

This is the classic GPU failure mode:

```text
parallel launch, serialized updates
```

---

## 5. Device-Side Compaction

After mask generation, the next step was to compact selected rows on the GPU.

The compaction pipeline was:

```text
filter -> mask -> prefix scan -> scatter selected indices
```

The purpose was to produce a compacted list of selected row indices:

```text
selected_indices[]
```

This allowed later stages to operate only on selected rows without transferring intermediate data back to the CPU.

### Compaction Stages

| Stage | Purpose |
|---|---|
| Filter | produce 0/1 mask |
| Prefix scan | compute output positions |
| Scatter | write selected row indices |
| Reduction/GROUP BY | process compacted selected rows |

---

## 6. D2H Transfer Lesson

An early version copied compacted indices back to the CPU for validation.

For 50M rows, the selected count was about 17.2M rows. Since compacted indices used `uint32_t`, the D2H payload was about 69 MB.

That D2H transfer dominated runtime when using pageable host memory and synchronous copy.

This showed that compaction is not primarily valuable as a host-transfer optimization when selectivity is high.

At ~34% selectivity:

```text
compacted uint32 indices = selected_count * 4 bytes
byte mask = row_count * 1 byte
```

Since:

```text
0.34 * 4 bytes > 1 byte
```

the compacted index representation can be larger than a byte mask.

### Key Observation

> The real value of compaction is not necessarily reducing D2H size. Its value is creating a device-resident representation that can feed downstream GPU stages.

---

## 7. GPU-Only SUM Over Selected Rows

The pipeline was then modified so that compacted indices remained on the GPU.

The GPU performed SUM directly over selected rows, and only the final scalar result was copied back to the CPU.

This eliminated the large intermediate D2H transfer.

The pipeline became:

```text
filter -> scan -> scatter -> reduce -> final D2H result
```

D2H time dropped from tens of milliseconds to near-zero scalar transfer cost.

---

## 8. Reduction Strategies

Two reduction strategies were implemented.

### 8.1 Atomic Reduction

The atomic reduction baseline performed:

```cpp
atomicAdd(global_sum, fare_amount[selected_indices[i]]);
```

This was simple, but all selected rows updated the same global accumulator.

### 8.2 Block-Partial Reduction

The block-partial approach used hierarchical aggregation:

1. each block processes a subset of selected rows
2. each block computes a partial sum locally
3. partial results are merged into the final result

This reduced the number of global atomic updates dramatically.

---

## 9. Reduction Results

For the 50M-row dataset:

| Algorithm | Reduce ms | Approx Throughput |
|---|---:|---:|
| Atomic reduction | ~60 ms | ~1.1 GB/s |
| Block-partial reduction | ~4.3 ms | ~15.9 GB/s |

The block-partial strategy achieved roughly a 14x reduction-stage speedup.

### Interpretation

Atomic reduction throughput stayed nearly flat as dataset size increased, suggesting it was capped by contention.

Block-partial reduction had higher setup cost at small scale, but scaled much better once selected-row count became large enough to amortize the extra structure.

### Key Observation

> Atomic reduction can look acceptable at small scale, but hierarchical reduction becomes much better at large scale.

---

## 10. Adding `passenger_count` for GROUP BY

To support GROUP BY, the data pipeline was extended to include `passenger_count`.

The SoA layout became:

```text
trip_distance[]
fare_amount[]
passenger_count[]
```

The binary format and CSV ingestion path were updated to include the new column.

A CPU reference GROUP BY implementation was also added for correctness validation.

Passenger count cardinality was low, typically around 7–9 keys depending on dataset and generated distribution.

---

## 11. GROUP BY Implementations

Two GROUP BY strategies were implemented.

### 11.1 Naive Atomic GROUP BY

The naive implementation directly updated global per-key accumulators:

```cpp
atomicAdd(group_sum[key], fare_amount[i]);
atomicAdd(group_count[key], 1);
```

This was simple, but heavily sensitive to key distribution.

When many selected rows mapped to the same key, they contended on the same global memory bucket.

### 11.2 Compact + Block-Partial GROUP BY

The block-partial GROUP BY implementation used the compacted selected row list and local aggregation before global merge.

The pipeline was:

```text
filter
-> scan
-> scatter selected rows
-> block-local per-key aggregation
-> global merge
```

Because `passenger_count` has small cardinality, the block-local aggregation could use a small fixed key range.

This reduced the number of global atomic operations and made performance less sensitive to key skew.

---

## 12. GROUP BY Scaling Results

For the 50M-row dataset:

| Algorithm | GROUP BY ms | Total GPU ms |
|---|---:|---:|
| Naive atomic GROUP BY | ~66 ms | ~103 ms |
| Block-partial GROUP BY | ~11 ms | ~59 ms |

The block-partial GROUP BY stage was about 6x faster, and the total GPU pipeline was about 1.7x faster.

### Interpretation

The block-partial strategy paid extra overhead for filtering, scan, and scatter, but it significantly reduced contention in the GROUP BY stage.

At small dataset sizes, the atomic implementation could remain competitive because it had less orchestration overhead.

At larger dataset sizes, contention dominated, and block-partial aggregation became clearly better.

---

## 13. Key Distribution Stress Test

To test contention sensitivity, three synthetic key distributions were evaluated:

| Distribution | Description |
|---|---|
| Uniform | keys spread evenly |
| Zipf | power-law skew |
| Hotkey | most rows map to one key |

The goal was to understand how key skew affects atomic and block-partial aggregation.

---

## 14. Distribution Results

For 50M rows:

| Distribution | Atomic GROUP BY ms | Block-Partial GROUP BY ms | Speedup |
|---|---:|---:|---:|
| Uniform | ~32 ms | ~13 ms | ~2.5x |
| Zipf | ~43 ms | ~11 ms | ~4.0x |
| Hotkey | ~59 ms | ~7 ms | ~8.4x |

All correctness checks passed.

---

## 15. Distribution Analysis

The atomic implementation was highly sensitive to distribution.

As skew increased, more threads updated fewer global buckets. This increased contention and forced serialization in the memory subsystem.

The block-partial implementation was more stable because it performed local aggregation before global merge.

Interestingly, the hotkey distribution was fastest for the block-partial implementation. This makes sense because fewer active keys reduce local aggregation complexity, while the global merge remains bounded by the number of blocks rather than the number of selected rows.

### Key Observation

> Naive atomic GROUP BY gets worse as key skew increases, while block-partial aggregation benefits from reducing global update frequency.

---

## 16. What Scaled Well

The following parts scaled well:

- mask-only filtering
- prefix scan and scatter at large dataset sizes
- block-partial reduction
- block-partial GROUP BY under skewed distributions

The best-performing stages were those that avoided global contention and kept work local before merging.

---

## 17. What Did Not Scale Well

The following parts did not scale well:

- global atomic filter+sum
- global atomic selected-row reduction
- naive atomic GROUP BY under skew
- copying intermediate selected indices back to CPU

These designs shared the same structural issue:

```text
many parallel threads converging onto a small number of global targets
```

---

## 18. Core Systems Lessons

### 18.1 Parallelism Alone Is Not Enough

Launching many GPU threads does not guarantee scalable performance.

If many threads update the same global address, the memory system serializes progress.

### 18.2 Memory Movement Must Be Designed

Large intermediate transfers over PCIe can dominate runtime.

Intermediate data should remain device-resident whenever downstream GPU stages can consume it.

### 18.3 Key Distribution Matters

GROUP BY performance depends strongly on key distribution.

Uniform keys reduce contention.

Skewed keys concentrate contention.

Hotkey workloads are especially damaging for naive global atomic algorithms.

### 18.4 Hierarchical Aggregation Improves Scalability

Block-local aggregation reduces global traffic and limits contention.

This is especially useful when key cardinality is small or distributions are skewed.

### 18.5 Abstraction Helps When It Preserves Structure

The filter -> scan -> scatter pipeline is more complex than a fused atomic kernel, but it exposes better control over data movement and aggregation strategy.

---

## 19. Month 2 Outcome

By the end of Month 2, G-Query Light evolved from:

```text
single GPU kernel with global atomics
```

into:

```text
GPU-native columnar query pipeline
with filtering, compaction, reduction, GROUP BY,
contention analysis, and distribution stress testing
```

The project now demonstrates:

- understanding of GPU memory behavior
- atomic contention diagnosis
- hierarchical reduction design
- low-cardinality GROUP BY handling
- benchmark-driven performance reasoning
- correctness validation against CPU reference

---

## 20. Remaining Limitations

This repo is still primarily an exploration repo.

Known limitations:

- code organization is rough
- multiple experimental paths coexist
- benchmark harness is functional but not polished
- algorithms are educational rather than production-grade
- no deep Nsight Systems or Nsight Compute analysis yet
- no stream overlap yet
- no warp-level aggregation yet
- no CUB/Thrust GROUP BY comparison yet

This is acceptable for the first repo because its role is to capture experimentation and learning.

A cleaner follow-up repo should use a more deliberate structure.

---

## 21. Next Direction: Month 3

Month 3 should shift focus from algorithm implementation to hardware observability and execution timelines.

Potential focus areas:

- Nsight Systems timeline analysis
- Nsight Compute kernel-level analysis
- NVTX ranges
- CUDA streams
- overlapping H2D transfers with compute
- pinned memory and async copy behavior
- identifying idle gaps
- measuring kernel launch overhead
- comparing fused vs staged pipelines
- evaluating when intermediate materialization is worth the cost

The Month 3 goal should be:

```text
understand where the GPU is idle,
why it is idle,
and how to structure the pipeline to reduce idle time
```

---

## Final Summary

Month 2 established the core performance lesson of this project:

> scalable GPU query execution is not mainly about launching more threads;
> it is about controlling memory movement, synchronization, and contention.

The most important result was showing that naive global atomic aggregation is simple but fragile, while hierarchical block-partial aggregation is more robust under large datasets and skewed key distributions.

This provides a strong foundation for Month 3 hardware observability work.
````
