# G-Query Light — Phase 1: The Memory Wall

## Goal

G-Query Light is a CLI-based C++/CUDA data engine built to study host-device orchestration and PCIe bottlenecks.

Phase 1 focuses on a simple analytical workload:
- load a large NYC Taxi CSV dataset
- extract a small set of numeric columns
- apply a fixed filter predicate
- compute a count and a sum
- establish a CPU baseline before GPU offload

This phase is intentionally narrow. The goal is not query-language flexibility or advanced kernel optimization. The goal is disciplined measurement of data movement and execution time.

---

## Fixed Phase 1 workload

### Dataset
NYC Taxi CSV files

### Columns used
- `trip_distance`
- `fare_amount`

### Predicate
```cpp
trip_distance > 2.5f && fare_amount > 10.0f
```

### Output
- count
- sum_fare_amount

Sum is accumulated in double for numerical stability.