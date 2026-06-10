# TRmorph MorphBPE 32k UHeM Operations Report

Generated at: `2026-06-10T10:12:32.049619+00:00`

## Job Chain

- Segmentation array: `493352`
- Finalize/tokenizer job: `493359`
- Single-node A100 model job: `493360`
- HF tokenizer publish job: `493361`

Current Slurm queue snapshot:

```text
JOBID PARTITION                                 NAME    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            493360   a100x4q              nanochat-tr-mbpe-trm32k  PENDING       0:00 10-00:00:00      1 (Dependency)
  493352_[23-29%6]    cpu2dq              nanochat-seg-trmorph32k  PENDING       0:00 2-00:00:00      1 (JobArrayTaskLimit)
            493361    cpu2dq                nanochat-tokenizer-hf  PENDING       0:00   4:00:00      1 (Dependency)
            493359    cpu2dq        nanochat-finalize-mbpe-trm32k  PENDING       0:00 1-00:00:00      1 (Dependency)
         493352_22    cpu2dq              nanochat-seg-trmorph32k  RUNNING       8:29 2-00:00:00      1 a041
         493352_21    cpu2dq              nanochat-seg-trmorph32k  RUNNING      30:57 2-00:00:00      1 a072
         493352_20    cpu2dq              nanochat-seg-trmorph32k  RUNNING      33:57 2-00:00:00      1 a066
         493352_19    cpu2dq              nanochat-seg-trmorph32k  RUNNING      35:57 2-00:00:00      1 a079
         493352_18    cpu2dq              nanochat-seg-trmorph32k  RUNNING    1:02:57 2-00:00:00      1 a073
         493352_16    cpu2dq              nanochat-seg-trmorph32k  RUNNING    5:10:29 2-00:00:00      1 a083
```

## Segmentation Progress

- Completed shard sidecars: `17`
- Completed segmented parquet files: `17`
- Active temp shard files: `6`
- Final segmentation manifest exists: `False`
- Final dataset manifest exists: `False`

Completed-shard content totals:

- Documents: `58,477,000`
- Word tokens seen by segmenter: `21,758,849,352`
- Split words: `8,081,788,036` (37.14%)
- Fallback words: `3,706,717,293` (17.04%)

Shard wall-time stats from `.done.json` sidecars:

- Sum over completed shards: `92.35` shard-hours
- Mean per completed shard: `5.43` hours
- Median per completed shard: `5.40` hours
- Max completed shard: `5.96` hours

## UHeM Resource Accounting

Slurm accounting is reported two ways:

- Allocated CPU-hours: `Elapsed * AllocCPUS`; this is the conservative cluster-allocation view.
- TotalCPU hours: Slurm's measured CPU time; this is the process-consumption view.

- Completed segmentation tasks: `17`
- Running segmentation tasks: `6`
- Pending segmentation tasks: `7`
- Failed/other segmentation tasks: `0`
- Completed allocated CPU-hours: `11837.55`
- Completed TotalCPU hours: `116.34`
- Running allocated CPU-hours so far: `1029.69`
- Running TotalCPU hours so far: `0.00`

## Slurm Task Table

| JobID | State | Elapsed | TotalCPU | AllocCPUS | Alloc CPU-hours | Node | Start | End |
|---|---:|---:|---:|---:|---:|---|---|---|
| `493352_0` | `COMPLETED` | `05:12:56` | `06:37:58` | `128` | `667.59` | `a080` | `2026-06-09T20:25:26` | `2026-06-10T01:38:22` |
| `493352_1` | `COMPLETED` | `05:57:59` | `07:35:30` | `128` | `763.70` | `a041` | `2026-06-09T20:25:26` | `2026-06-10T02:23:25` |
| `493352_2` | `COMPLETED` | `05:30:34` | `06:56:01` | `128` | `705.21` | `a043` | `2026-06-09T20:25:26` | `2026-06-10T01:56:00` |
| `493352_3` | `COMPLETED` | `05:56:40` | `07:24:53` | `128` | `760.89` | `a047` | `2026-06-09T20:25:26` | `2026-06-10T02:22:06` |
| `493352_4` | `COMPLETED` | `05:53:05` | `07:33:16` | `128` | `753.24` | `a050` | `2026-06-09T20:25:26` | `2026-06-10T02:18:31` |
| `493352_5` | `COMPLETED` | `05:57:28` | `07:33:42` | `128` | `762.60` | `a023` | `2026-06-09T20:25:26` | `2026-06-10T02:22:54` |
| `493352_6` | `COMPLETED` | `05:45:52` | `07:09:11` | `128` | `737.85` | `a083` | `2026-06-10T01:38:33` | `2026-06-10T07:24:25` |
| `493352_8` | `COMPLETED` | `05:10:08` | `06:40:46` | `128` | `661.62` | `a072` | `2026-06-10T02:03:33` | `2026-06-10T07:13:41` |
| `493352_9` | `COMPLETED` | `04:57:40` | `06:16:34` | `128` | `635.02` | `a079` | `2026-06-10T02:18:33` | `2026-06-10T07:16:13` |
| `493352_10` | `COMPLETED` | `05:48:04` | `07:10:58` | `128` | `742.54` | `a080` | `2026-06-10T02:22:33` | `2026-06-10T08:10:37` |
| `493352_11` | `COMPLETED` | `04:56:23` | `06:14:10` | `128` | `632.28` | `a043` | `2026-06-10T02:23:00` | `2026-06-10T07:19:23` |
| `493352_12` | `COMPLETED` | `05:38:24` | `06:57:25` | `128` | `721.92` | `a044` | `2026-06-10T02:23:33` | `2026-06-10T08:01:57` |
| `493352_13` | `COMPLETED` | `04:52:44` | `06:05:28` | `128` | `624.50` | `a079` | `2026-06-10T07:16:33` | `2026-06-10T12:09:17` |
| `493352_14` | `COMPLETED` | `05:16:53` | `06:37:20` | `128` | `676.02` | `a066` | `2026-06-10T07:19:33` | `2026-06-10T12:36:26` |
| `493352_15` | `COMPLETED` | `05:16:44` | `06:34:32` | `128` | `675.70` | `a041` | `2026-06-10T07:24:33` | `2026-06-10T12:41:17` |
| `493352_16` | `RUNNING` | `05:10:28` | `00:00:00` | `128` | `662.33` | `a083` | `2026-06-10T08:02:01` | `Unknown` |
| `493352_17` | `COMPLETED` | `04:52:57` | `06:08:58` | `128` | `624.96` | `a080` | `2026-06-10T08:11:01` | `2026-06-10T13:03:58` |
| `493352_18` | `RUNNING` | `01:02:56` | `00:00:00` | `128` | `134.26` | `a073` | `2026-06-10T12:09:33` | `Unknown` |
| `493352_19` | `RUNNING` | `00:35:56` | `00:00:00` | `128` | `76.66` | `a079` | `2026-06-10T12:36:33` | `Unknown` |
| `493352_20` | `RUNNING` | `00:33:56` | `00:00:00` | `128` | `72.39` | `a066` | `2026-06-10T12:38:33` | `Unknown` |
| `493352_21` | `RUNNING` | `00:30:56` | `00:00:00` | `128` | `65.99` | `a072` | `2026-06-10T12:41:33` | `Unknown` |
| `493352_22` | `RUNNING` | `00:08:28` | `00:00:00` | `128` | `18.06` | `a041` | `2026-06-10T13:04:01` | `Unknown` |
| `493359` | `PENDING` | `00:00:00` | `00:00:00` | `32` | `0.00` | `None assigned` | `Unknown` | `Unknown` |
| `493360` | `PENDING` | `00:00:00` | `00:00:00` | `64` | `0.00` | `None assigned` | `Unknown` | `Unknown` |
| `493361` | `PENDING` | `00:00:00` | `00:00:00` | `4` | `0.00` | `None assigned` | `Unknown` | `Unknown` |
| `493352_7` | `COMPLETED` | `05:24:20` | `06:43:48` | `128` | `691.91` | `a072` | `2026-06-10T07:14:01` | `2026-06-10T12:38:21` |

## Notes

- Tokenizer training has not started until the finalizer job leaves `PENDING`.
- The model run and HF tokenizer upload are dependency-held until the finalizer succeeds.
- Refresh this report by rerunning `scripts/report_uhem_tokenizer_jobs.py` with the same job IDs.
