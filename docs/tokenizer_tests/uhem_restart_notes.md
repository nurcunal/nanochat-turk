# UHeM Restart Notes

Operational notes for intentionally paused or canceled cluster work.

## 2026-06-11 tokenizer metrics slot release

Goal: start optimized full-corpus tokenizer metrics job `493906`
(`nanochat-tokenizer-metrics32k-opt`) without touching the baseline tokenizer
metrics job `493751`.

Actions taken around `2026-06-11 16:35-16:38 +03`:

- Canceled `493892_10` on `a025`.
- Canceled `493892_11` on `a027`.
- Slurm immediately backfilled those slots with `493892_14` on `a032` and
  `493892_15` on `a011`.
- Held pending TurkishDelight segmentation array tasks with `scontrol hold
  493892`; pending tasks `493892_[16-29]` moved to `JobHeldUser`.
- Canceled replacement tasks `493892_14` and `493892_15`.
- Optimized tokenizer metrics job `493906` then started on `a025` at
  `2026-06-11 16:38:01 +03`.

Canceled TurkishDelight segmentation tasks to restart later:

- `493892_10`
- `493892_11`
- `493892_14`
- `493892_15`

Original command:

```bash
/ari/users/nunal/nanochat-turk/runs/uhem_nakane_segment_morphbpe_tdelight_32k_array.sbatch
```

Restart plan after the optimized tokenizer metrics job is no longer blocking
CPU job slots:

```bash
ssh -o BatchMode=yes uhem-altay
cd /ari/users/nunal/nanochat-turk

# Release the held original pending tasks 16-29.
scontrol release 493892

# Restart the canceled tasks as a small replacement array.
sbatch --array=10,11,14,15%4 runs/uhem_nakane_segment_morphbpe_tdelight_32k_array.sbatch

# Check both the released original array and replacement array.
squeue -u "$USER" -o "%.18i %.10P %.35j %.2t %.12M %.6D %N %R"
```

Dependency note: `493858` (`nanochat-finalize-mbpe-tdlt32k`) was pending on the
TurkishDelight segmentation dependency at the time of cancellation. After the
replacement array finishes, verify whether `493858` is still valid; if Slurm
marks its dependency unsatisfied because tasks `10`, `11`, `14`, or `15` were
canceled in the original array, resubmit the finalize sbatch after all
segmentation shards are present.

## 2026-06-12 optimized tokenizer metrics retry

Job `493906` (`nanochat-tokenizer-metrics32k-opt`) confirmed that row-group
parallelism works, but the default was too aggressive:

- `num_workers=128`
- requested memory: `64G`
- failed after `00:19:59` with exit code `137`
- Slurm reported about `304 GB` memory utilized
- last progress line was around `85080 / 372849` row groups for the first
  tokenizer pass

The optimized sbatch was changed to a safer memory-bounded default:

- `--cpus-per-task=32`
- `--mem=128G`
- `NUM_WORKERS=${SLURM_CPUS_PER_TASK:-32}`
- `TOKENIZER_THREADS_PER_WORKER=1`

Do not infer a tokenizer failure from `493906`; it was an operational OOM.

## 2026-06-12 Zemberek reference-metric slot release

Goal: run the matched `50,000`-document Zemberek TRmorph-reference metric without
touching `a080` or the problematic baseline metrics job `493751`.

Actions taken:

- Held the pending tail of the TurkishDelight segmentation resume array:
  `494059_[20-29]`.
- Canceled exactly one running TurkishDelight task, `494059_19` on `a044`, to
  free a CPU node.
- Submitted `494080` (`nanochat-zemb50k-metrics`) with
  `runs/uhem_tokenizer_metric_zemberek_50k_trmorph_reference.sbatch`.
- The metric job ran on `a051` and completed successfully in `00:00:49`.
- Released the held TurkishDelight pending tail after the Zemberek metric job
  started; `494059_20` backfilled onto `a044`.

Canceled TurkishDelight segmentation task to restart later:

- `494059_19`

Restart command after the active TurkishDelight segmentation wave has room:

```bash
ssh -o BatchMode=yes uhem-altay
cd /ari/users/nunal/nanochat-turk
sbatch --array=19 runs/uhem_nakane_segment_morphbpe_tdelight_32k_array.sbatch
squeue -u "$USER" -o "%.18i %.10P %.35j %.2t %.12M %.6D %N %R"
```

Zemberek output archived from UHeM:

```text
/ari/users/nunal/nanochat-turk-morphbpe-zemberek-32768/report/tokenizer_metrics_50k_trmorph_reference/morphbpe_zemberek_32768_metrics.json
```
