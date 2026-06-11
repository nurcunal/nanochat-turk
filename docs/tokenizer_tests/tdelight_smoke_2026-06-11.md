# TurkishDelight Segmentation Smoke Tuning

Date: 2026-06-11, UHeM/Altay.

Goal: benchmark TurkishDelight segmentation settings before submitting the
remaining full FineWeb-2 Turkish shards. The original full-shard run used many
short-lived command subprocesses, which caused large process counts and disk
wait from repeated DyNet model loads.

## Completed Full Shards Before Tuning

Original long-running shards `0,1,2,5,6` completed before the optimized
remaining shards were submitted.

| Shard | Docs | Words | Elapsed s | Words/s |
|---|---:|---:|---:|---:|
| `000_00000` | 3,381,000 | 1,282,426,655 | 44,794.39 | 28,629.18 |
| `000_00001` | 3,396,000 | 1,281,805,857 | 44,894.21 | 28,551.70 |
| `000_00002` | 3,403,000 | 1,280,709,153 | 44,865.41 | 28,545.58 |
| `000_00005` | 3,406,000 | 1,281,394,129 | 44,844.35 | 28,574.26 |
| `001_00000` | 3,476,000 | 1,277,887,549 | 44,797.83 | 28,525.66 |

## Smoke Design

Smoke shard: raw shard index `3` (`000_00003.parquet`).

Smoke cap: `1024` documents.

Output root:

```text
/ari/users/nunal/nanochat-turk-morphbpe-tdelight-32768/smoke/tdelight-segmentation
```

All persistent runs set `TDELIGHT_PERSISTENT_COMMAND=1`, keeping one
TurkishDelight command process alive per worker so the DyNet model is loaded
once per worker rather than once per batch.

## First Sweep: Job 493763

| Config | Persistent | Workers | Segment Batch | Row Group Batch | Words/s | Elapsed s | Fallback Words |
|---|---:|---:|---:|---:|---:|---:|---:|
| `legacy_w16_b512_rg32` | 0 | 16 | 512 | 32 | 1,311.16 | 456.46 | 0 |
| `persistent_w16_b512_rg32` | 1 | 16 | 512 | 32 | 21,386.30 | 27.98 | 0 |
| `persistent_w32_b512_rg32` | 1 | 32 | 512 | 32 | 14,160.29 | 42.27 | 0 |
| `persistent_w64_b512_rg32` | 1 | 64 | 512 | 32 | 13,689.88 | 43.72 | 0 |
| `persistent_w32_b2048_rg32` | 1 | 32 | 2048 | 32 | 14,914.23 | 40.13 | 0 |
| `persistent_w64_b2048_rg32` | 1 | 64 | 2048 | 32 | 14,536.19 | 41.17 | 0 |

The first sweep showed that persistence is essential, and that too many
workers hurts throughput because of model/process contention.

## Second Sweep: Job 493843

| Config | Persistent | Workers | Segment Batch | Row Group Batch | Words/s | Elapsed s | Fallback Words |
|---|---:|---:|---:|---:|---:|---:|---:|
| `persistent_w4_b512_rg32` | 1 | 4 | 512 | 32 | 16,275.81 | 36.77 | 0 |
| `persistent_w8_b512_rg32` | 1 | 8 | 512 | 32 | 21,487.48 | 27.85 | 0 |
| `persistent_w12_b512_rg32` | 1 | 12 | 512 | 32 | 25,096.40 | 23.85 | 0 |
| `persistent_w16_b256_rg32` | 1 | 16 | 256 | 32 | 26,766.36 | 22.36 | 0 |
| `persistent_w16_b2048_rg32` | 1 | 16 | 2048 | 32 | 25,601.30 | 23.38 | 0 |
| `persistent_w24_b512_rg32` | 1 | 24 | 512 | 32 | 23,581.09 | 25.38 | 0 |

Best smoke setting: `persistent_w16_b256_rg32`.

## Full Remaining Shards

Remaining shards `3,4,7,8,9` were submitted as job array `493850` using the
best smoke setting:

```text
TDELIGHT_PERSISTENT_COMMAND=1
SEGMENT_WORKERS=16
SEGMENT_MAX_IN_FLIGHT=16
SEGMENT_BATCH_SIZE=256
ROW_GROUP_BATCH_SIZE=32
SEGMENT_TIMEOUT=900
```

Early process sampling showed about `35` TurkishDelight processes per node,
compared with about `258` per node in the original full-shard runs. Early
parquet growth was about `21-28 MB/min` per shard.

## Resource Note

The first five completed full shards used `128` Slurm CPUs and `128` process
workers per shard. Shards `000_00001` and `000_00002` finished in
`44,894.21` seconds and `44,865.41` seconds respectively, about `12.47` hours
per shard and about `28.55k` words/s.

The persistent TurkishDelight path was selected because smoke tests showed it
avoids repeated DyNet model loads. Its best smoke setting used `16` workers at
`26.77k` words/s on a 1024-document sample. Extrapolated to a full
approximately `1.28B`-word shard, that is roughly `13.3` hours per shard. This
is not a clear wall-clock speedup over the earlier `128`-worker full shards;
it is primarily a resource-efficiency improvement when the Slurm CPU request
is also reduced.

The submitted remaining-shard job `493850` still requested `128` CPUs per
array task while using `16` workers, so it is healthy but not cost-optimal.
Future submissions should keep `SEGMENT_WORKERS` aligned with
`SLURM_CPUS_PER_TASK` and scale through array concurrency. The UHeM array
script now defaults to `16` CPUs, `16` workers, and higher array concurrency
for that reason.
