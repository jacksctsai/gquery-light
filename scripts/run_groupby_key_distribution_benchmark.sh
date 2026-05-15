#!/usr/bin/env bash
# Run groupby atomic vs block_partial on 50M key-distribution binaries and print comparable CSV lines.
# Usage: ./scripts/run_groupby_key_distribution_benchmark.sh [path/to/gquery] [iterations] [threads_per_block]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GQUERY="${1:-${ROOT}/build/gquery}"
ITERS="${2:-5}"
TPB="${3:-256}"
DATA_DIR="${ROOT}/data/nyc_yellow"

if [[ ! -x "$GQUERY" ]]; then
  echo "error: gquery not found or not executable: $GQUERY" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "warning: no working NVIDIA driver; CUDA benchmarks will likely fail." >&2
fi

bins=(
  "${DATA_DIR}/yellow_50M_uniform.bin"
  "${DATA_DIR}/yellow_50M_hotkey.bin"
  "${DATA_DIR}/yellow_50M_zipf.bin"
)

labels=(uniform hotkey zipf)

echo "gquery=$GQUERY iterations=$ITERS threads_per_block=$TPB"
echo "---- per-dataset: mode= (metrics) and all_keys_correct ----"

for i in "${!bins[@]}"; do
  bin="${bins[$i]}"
  lab="${labels[$i]}"
  if [[ ! -f "$bin" ]]; then
    echo "skip ($lab): missing $bin" >&2
    continue
  fi
  echo ""
  echo "### ${lab} :: atomic"
  "$GQUERY" runbin_gpu_groupby_atomic "$bin" "$ITERS" "$TPB" | grep -E '^(mode=|all_keys_correct:)' || true
  echo "### ${lab} :: block_partial"
  "$GQUERY" runbin_gpu_groupby_block_partial "$bin" "$ITERS" "$TPB" | grep -E '^(mode=|all_keys_correct:)' || true
done
