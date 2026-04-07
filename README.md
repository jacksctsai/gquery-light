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

---

## Building

Requirements:

- **CMake** 3.5 or newer
- A **C++20** compiler
- **Optional:** an NVIDIA toolkit with `nvcc` on `PATH`. If CMake finds a CUDA compiler, the project defines `GQUERY_USE_CUDA` and links the GPU filter; otherwise `gquery` is CPU-only.

Configure and build (out-of-tree build recommended):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Smoke tests (no GPU required for the CPU paths):

```bash
./build/test_smoke
```

### CUDA: choosing the GPU architecture (`GQUERY_CUDA_ARCH`)

GPU code must be compiled for the **same virtual architecture** as the GPU you run on. CMake passes this through [`CUDA_ARCHITECTURES`](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html) via cache variable **`GQUERY_CUDA_ARCH`**.

- **Default:** `80` (NVIDIA Ampere, e.g. A100, RTX 30xx in many setups). This keeps **headless CI and machines without a visible GPU** from failing at configure time (unlike `native`, which needs device detection).
- **Your machine:** set `GQUERY_CUDA_ARCH` to the **without-decimal** form of your GPU’s **compute capability** (major×10 + minor). Examples:
  - Compute capability **7.5** (e.g. T4, RTX 20xx) → **`75`**
  - **8.0** → **`80`**
  - **8.6** → **`86`**
  - **8.9** (e.g. L4, RTX 40xx) → **`89`**

You can read compute capability from the driver, for example:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

Then configure once, for example:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGQUERY_CUDA_ARCH=75
cmake --build build -j
```

If the architecture does not match the GPU, `runbin` may fail at launch with an error similar to **“no kernel image is available for execution on the device.”** Reconfigure with the correct `GQUERY_CUDA_ARCH` and rebuild.

**`native`:** On a workstation where CMake can detect the installed GPU at configure time, you may use `-DGQUERY_CUDA_ARCH=native` so the build targets that GPU automatically. This is convenient locally but is a poor default for reproducible or CI builds when no GPU is present.

---

## CLI usage

### `csv2bin` — CSV → binary columnar file

```bash
./build/gquery csv2bin <input.csv> <output.bin>
```

Writes a small header (`uint64_t` row count) followed by raw `float` arrays: `trip_distance[]`, then `fare_amount[]`.

### `runbin` — load binary and run the filter

```bash
./build/gquery runbin <input.bin> [iterations]
```

- **`iterations`** defaults to `1`. It controls how many times the filter is run (useful for timing); aggregate **count** / **sum_fare_amount** are from the last run.

**With CUDA enabled** (`GQUERY_USE_CUDA`): each iteration allocates device buffers, copies columns host→device, runs a single simple kernel (one thread per row, atomics for count/sum), and copies results back. Output is **GPU mode** and does **not** print CPU-only fields like `filter_ms (avg)` unless the program explicitly measured CPU filtering in that same invocation. GPU mode prints `h2d_ms`, `kernel_ms`, `d2h_ms`, `total_gpu_ms`, `effective_h2d_gb_per_s` along with the common context fields (`file_bytes`, `payload_bytes`, `rows`, `iterations`, `load_ms`, `total_ms`, `count`, `sum_fare_amount`, `load_gb_per_s`) and a `mode=gpu,...` machine-readable line.

**Without CUDA:** the same predicate runs on the CPU; **`filter_ms (avg)`** is the per-iteration CPU filter time.

### Direct CSV path (CPU)

```bash
./build/gquery <input.csv> [iterations]
```

Parses CSV once, then repeats the CPU filter `iterations` times.

---

## Test data

`data/test_5_rows.csv` is used by `test_smoke` for the CSV reader. After `csv2bin`, the filter on those five rows should yield **count = 3** and **sum_fare_amount = 120** (rows with `trip_distance > 2.5` and `fare_amount > 10`; the row with `trip_distance == 2.5` does not qualify).