
# G-Query Light — Month 1 Checkpoint

## Phase 1: The Memory Wall (Host–Device Orchestration)

## Environment

* **Machine**: Google Cloud VM
* **GPU**: NVIDIA Tesla T4 (16 GB)
* **Driver**: 570.172.08
* **CUDA**: 12.8
* **CPU baseline**: same machine, single-threaded C++ implementation

---

## Dataset

* **Source**: NYC Yellow Taxi dataset
* **Rows**: 38,310,226
* **Schema (used columns)**:

  * `trip_distance` (float32)
  * `fare_amount` (float32)
* **Binary format**: SoA (Structure of Arrays)
* **Payload size**: 306,481,808 bytes (~306 MB)

---

## Workload

### Predicate

```cpp
trip_distance > 2.5f && fare_amount > 10.0f
```

### Output

* `count`
* `sum_fare_amount` (double accumulation)

---

## CPU Baseline (Binary Path)

| Metric      | Value     |
| ----------- | --------- |
| Payload     | ~306 MB   |
| Filter time | ~170 ms   |
| Throughput  | ~1.8 GB/s |

### Observation

* CSV parsing removed from hot path
* Workload becomes **memory-bound scan**
* CPU performance limited by memory bandwidth and access pattern

---

## GPU Baseline (Naive CUDA)

### Breakdown

| Stage         | Time         |
| ------------- | ------------ |
| H→D           | 24.84 ms     |
| Kernel        | 47.93 ms     |
| D→H           | 0.06 ms      |
| **Total GPU** | **72.84 ms** |

### Throughput

| Metric                  | Value       |
| ----------------------- | ----------- |
| Effective H→D bandwidth | ~12.34 GB/s |

---

## Correctness Validation

| Metric      | Result                     |
| ----------- | -------------------------- |
| Count match | ✅ exact                    |
| Sum match   | ✅ exact (within tolerance) |

---

## Performance Comparison

| System                     | Time     |
| -------------------------- | -------- |
| CPU (scan only)            | ~170 ms  |
| GPU (total incl. transfer) | ~72.8 ms |

### Speedup

* **~2.3× faster than CPU**

---

## Bottleneck Analysis

### Before (initial CSV pipeline)

* **Dominant cost**: CSV parsing (~99%)
* GPU offload ineffective

### After (binary SoA path)

* **Dominant costs**:

  * Kernel (~48 ms)
  * H→D transfer (~25 ms)

### Current classification

| Component | Status                       |
| --------- | ---------------------------- |
| PCIe      | significant but not dominant |
| Kernel    | dominant stage               |
| Compute   | trivial                      |
| Memory    | likely limiting factor       |

---

## Key Findings

### 1. System structure determines usefulness of GPU

* Initial pipeline: GPU acceleration irrelevant
* After isolating numeric stage: GPU becomes effective

### 2. Data layout matters

* SoA enabled:

  * contiguous memory access
  * clean PCIe transfer model
  * column-level reasoning

### 3. Transfer path quality matters

* Pageable memory → ~4.6 GB/s (initial)
* Improved path → ~12.3 GB/s
* Transfer optimization shifted bottleneck to kernel

### 4. GPU is not automatically compute-bound

* Kernel dominated by:

  * memory access
  * atomic contention
* not arithmetic complexity

---

## What Surprised Me

* CSV parsing dominated far more than expected
* GPU offload only became meaningful after removing text ingestion
* Transfer bandwidth varied significantly depending on host memory handling

---

## Initial Misconceptions

* “GPU is faster” → irrelevant without system-level measurement
* Dataset size alone does not guarantee PCIe efficiency
* Kernel optimization is useless if the wrong stage dominates

---

## Architectural Lessons

* Separate ingestion from computation
* Use columnar (SoA) layout for GPU workloads
* Always measure:

  * transfer
  * compute
  * total system time
* Optimize the dominant stage, not the most interesting one

---

## Next Steps (Month 2)

Focus: **Parallel Computation & Reduction Efficiency**

Planned improvements:

* Replace global atomics with:

  * block-level reduction
  * shared memory accumulation
* Introduce:

  * warp-level primitives
  * hierarchical reduction
* Re-measure:

  * kernel time
  * total GPU time
* Evaluate:

  * memory vs compute vs synchronization bottlenecks

---

## One-line Summary

> By restructuring the pipeline to isolate numeric processing and measuring host-device transfer explicitly, I demonstrated a 2.3× GPU speedup on a 306 MB dataset and identified the shift from PCIe-bound to kernel-dominated execution.
