# TRmorph MorphBPE 32k UHeM Operations Report

Generated at: `2026-06-10T19:37:24.057508+00:00`

## Job Chain

- Segmentation array: `493352`
- Finalize/tokenizer job: `493359`
- Single-node A100 model job: `493360`
- HF tokenizer publish job: `493361`

Current Slurm queue snapshot:

```text
JOBID PARTITION                                 NAME    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
```

## Segmentation Progress

- Completed shard sidecars: `30`
- Completed segmented parquet files: `30`
- Active temp shard files: `0`
- Final segmentation manifest exists: `True`
- Final dataset manifest exists: `True`

Completed-shard content totals:

- Documents: `95,129,129`
- Word tokens seen by segmenter: `35,676,332,563`
- Split words: `13,115,243,179` (36.76%)
- Fallback words: `6,205,142,242` (17.39%)

Shard wall-time stats from `.done.json` sidecars:

- Sum over completed shards: `148.61` shard-hours
- Mean per completed shard: `4.95` hours
- Median per completed shard: `5.21` hours
- Max completed shard: `5.97` hours

## UHeM Resource Accounting

Slurm accounting is reported two ways:

- Allocated CPU-hours: `Elapsed * AllocCPUS`; this is the conservative cluster-allocation view.
- TotalCPU hours: Slurm's measured CPU time; this is the process-consumption view.

- Completed segmentation tasks: `30`
- Running segmentation tasks: `0`
- Pending segmentation tasks: `0`
- Failed/other segmentation tasks: `0`
- Completed allocated CPU-hours: `19050.24`
- Completed TotalCPU hours: `186.62`
- Running allocated CPU-hours so far: `0.00`
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
| `493352_16` | `COMPLETED` | `05:15:07` | `06:34:05` | `128` | `672.25` | `a083` | `2026-06-10T08:02:01` | `2026-06-10T13:17:08` |
| `493352_17` | `COMPLETED` | `04:52:57` | `06:08:58` | `128` | `624.96` | `a080` | `2026-06-10T08:11:01` | `2026-06-10T13:03:58` |
| `493352_18` | `COMPLETED` | `05:58:23` | `07:20:09` | `128` | `764.55` | `a073` | `2026-06-10T12:09:33` | `2026-06-10T18:07:56` |
| `493352_19` | `COMPLETED` | `04:50:50` | `06:06:18` | `128` | `620.44` | `a079` | `2026-06-10T12:36:33` | `2026-06-10T17:27:23` |
| `493352_20` | `COMPLETED` | `05:13:44` | `06:32:04` | `128` | `669.30` | `a066` | `2026-06-10T12:38:33` | `2026-06-10T17:52:17` |
| `493352_21` | `COMPLETED` | `04:49:27` | `06:02:08` | `128` | `617.49` | `a072` | `2026-06-10T12:41:33` | `2026-06-10T17:31:00` |
| `493352_22` | `COMPLETED` | `05:13:02` | `06:30:17` | `128` | `667.80` | `a041` | `2026-06-10T13:04:01` | `2026-06-10T18:17:03` |
| `493352_23` | `COMPLETED` | `05:11:11` | `06:27:05` | `128` | `663.86` | `a083` | `2026-06-10T13:17:33` | `2026-06-10T18:28:44` |
| `493352_24` | `COMPLETED` | `04:01:07` | `04:58:19` | `128` | `514.38` | `a062` | `2026-06-10T15:36:06` | `2026-06-10T19:37:13` |
| `493352_25` | `COMPLETED` | `03:16:09` | `04:07:56` | `128` | `418.45` | `a021` | `2026-06-10T15:36:06` | `2026-06-10T18:52:15` |
| `493352_26` | `COMPLETED` | `03:29:31` | `04:15:32` | `128` | `446.97` | `a022` | `2026-06-10T15:36:06` | `2026-06-10T19:05:37` |
| `493352_27` | `COMPLETED` | `03:01:20` | `03:46:14` | `128` | `386.84` | `a023` | `2026-06-10T15:36:06` | `2026-06-10T18:37:26` |
| `493352_28` | `COMPLETED` | `03:17:11` | `04:12:07` | `128` | `420.66` | `a024` | `2026-06-10T15:36:06` | `2026-06-10T18:53:17` |
| `493352_29` | `COMPLETED` | `02:43:55` | `03:24:39` | `128` | `349.69` | `a025` | `2026-06-10T15:36:06` | `2026-06-10T18:20:01` |
| `493359` | `COMPLETED` | `01:29:31` | `08:09:41` | `128` | `190.97` | `a083` | `2026-06-10T19:37:34` | `2026-06-10T21:07:05` |
| `493360` | `FAILED` | `00:00:03` | `00:00.272` | `64` | `0.05` | `a146` | `2026-06-10T21:07:34` | `2026-06-10T21:07:37` |
| `493361` | `FAILED` | `00:00:02` | `00:00.267` | `128` | `0.07` | `a083` | `2026-06-10T21:07:34` | `2026-06-10T21:07:36` |
| `493352_7` | `COMPLETED` | `05:24:20` | `06:43:48` | `128` | `691.91` | `a072` | `2026-06-10T07:14:01` | `2026-06-10T12:38:21` |

## Notes

- Finalize/tokenizer prep completed successfully.
- Model run state: `FAILED`.
- HF tokenizer upload state: `FAILED`.
- Failed downstream jobs should be diagnosed from their Slurm logs and resubmitted after wrapper fixes; segmented shards and tokenizer artifacts are preserved.
- Refresh this report by rerunning `scripts/report_uhem_tokenizer_jobs.py` with the same job IDs.
