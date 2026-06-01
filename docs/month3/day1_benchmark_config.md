# Month 3 Day 1 — Benchmark Config

## Goal

Make G-Query Light benchmark configurations reproducible from CLI without source edits.

## Acceptance Runs

| Rows | Distribution | Mode | Validate | Avg GPU ms | Data Gen ms |
|---:|---|---|---|---:|---:|
| 1,000,000 | uniform | atomic | true | 2.9469 | 2511.0231 |
| 12,000,000 | hotkey | block-partial | false | 24.4566 | 3748.9956 |
| 64,000,000 | uniform | block-partial | false | 83.6174 | 6700.5877 |

## Notes

- CPU synthetic data generation runs once before benchmark iterations.
- Warmup iterations are excluded from timing.
- Validation can be disabled for profiling.
- No source edits are required between benchmark configurations.

## Status

Day 1 acceptance criteria met.