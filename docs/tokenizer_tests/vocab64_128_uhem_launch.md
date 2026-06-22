# 64k/128k Tokenizer Ablation Launch

Date: 2026-06-12

This note records the larger-vocabulary tokenizer and model-training launch for
the Turkish MorphBPE ablation.

## Grid

| Vocab | Depth | Tokenizer family | Segmenter | Tokenizer name | Model tag |
| ---: | ---: | --- | --- | --- | --- |
| 64k | d16 | raw BPE | none | `bpe_64k` | `tr_d16_bpe_64k` |
| 64k | d16 | MorphBPE | TRmorph | `morphbpe_trmorph_64k` | `tr_d16_morphbpe_trmorph_64k` |
| 64k | d16 | MorphBPE | Zemberek | `morphbpe_zemberek_64k` | `tr_d16_morphbpe_zemberek_64k` |
| 64k | d16 | MorphBPE | TurkishDelightNLP | `morphbpe_tdelight_64k` | `tr_d16_morphbpe_tdelight_64k` |
| 128k | d12 | raw BPE | none | `bpe_128k` | `tr_d12_bpe_128k` |
| 128k | d12 | MorphBPE | TRmorph | `morphbpe_trmorph_128k` | `tr_d12_morphbpe_trmorph_128k` |
| 128k | d12 | MorphBPE | Zemberek | `morphbpe_zemberek_128k` | `tr_d12_morphbpe_zemberek_128k` |
| 128k | d12 | MorphBPE | TurkishDelightNLP | `morphbpe_tdelight_128k` | `tr_d12_morphbpe_tdelight_128k` |

## Launchers

- Tokenizer finalizer:
  [`runs/uhem_nakane_finalize_tokenizer_ablation.sbatch`](../../runs/uhem_nakane_finalize_tokenizer_ablation.sbatch)
- A100x4 model trainer:
  [`runs/uhem_nakane_a100x4_tokenizer_ablation.sbatch`](../../runs/uhem_nakane_a100x4_tokenizer_ablation.sbatch)
- Grid submitter:
  [`runs/uhem_submit_64k_128k_tokenizer_ablation.sh`](../../runs/uhem_submit_64k_128k_tokenizer_ablation.sh)

The submitter creates one CPU tokenizer job per grid row and one dependent
A100x4 model-training job per tokenizer job. GPU training keeps
`TOTAL_BATCH_SIZE=1048576` and `NUM_ITERATIONS=17100`, matching the 32k d20
token-position budget. This gives the same `17.93B` training token positions
while using d16 for 64k and d12 for 128k to keep total parameters near the
current 1B budget.

Tokenizer finalizer jobs write compact GitHub-ready bundles under:

```text
artifacts/tokenizers/<tokenizer_name>/
```

The large raw and segmented corpora stay on UHeM. The final bundles should be
copied back and committed after the corresponding tokenizer jobs complete.

## Submitted Jobs

Submission was prepared locally and pushed to GitHub in commit `e947763`.
Intermittent SSH hostname resolution failures delayed the live submission, so
the launchers were fetched on UHeM from GitHub raw URLs once SSH recovered.

The intended one-shot command is:

```bash
cd /ari/users/nunal/nanochat-turk
TDELIGHT_READY_DEPENDENCY=494159 \
  bash runs/uhem_submit_64k_128k_tokenizer_ablation.sh | tee vocab64_128_submit_$(date +%Y%m%d_%H%M%S).tsv
```

Use `TDELIGHT_READY_DEPENDENCY=494159` because job `494159`
(`nanochat-finalize-mbpe-tdlt32k-v2`) is the existing TurkishDelight finalizer
queued behind the last TurkishDelight segmentation shard. The 64k/128k
TurkishDelight tokenizer jobs should wait for it so they train on the complete
segmented corpus.

The first submitter version did create tokenizer jobs, but UHeM prints a billing
notice before the numeric `sbatch --parsable` job id. That polluted the captured
dependency string, so the dependent GPU jobs were submitted manually after
patching the submitter to parse the last numeric job id from `sbatch` output.

| Vocab | Tokenizer family | Segmenter | Tokenizer job | Train job | State at submission |
| ---: | --- | --- | ---: | ---: | --- |
| 64k | raw BPE | none | `494181` | `494189` | tokenizer running; train waits on tokenizer |
| 64k | MorphBPE | TRmorph | `494182` | `494190` | tokenizer running; train waits on tokenizer |
| 64k | MorphBPE | Zemberek | `494183` | `494191` | tokenizer running; train waits on tokenizer |
| 64k | MorphBPE | TurkishDelightNLP | `494184` | `494192` | tokenizer waits on `494159`; train waits on tokenizer |
| 128k | raw BPE | none | `494185` | `494193` | tokenizer running; train waits on tokenizer |
| 128k | MorphBPE | TRmorph | `494186` | `494194` | tokenizer running; train waits on tokenizer |
| 128k | MorphBPE | Zemberek | `494187` | `494195` | tokenizer running; train waits on tokenizer |
| 128k | MorphBPE | TurkishDelightNLP | `494188` | `494196` | tokenizer waits on `494159`; train waits on tokenizer |

## 2026-06-12 Tokenizer Completion Check

Raw BPE tokenizer jobs completed:

| Tokenizer | Job | State | Raw metric bundle |
| --- | ---: | --- | --- |
| `bpe_64k` | `494181` | `COMPLETED`, exit `0:0` | `/ari/users/nunal/nanochat-turk/artifacts/tokenizers/bpe_65536/metrics/raw_metrics.json` |
| `bpe_128k` | `494185` | `COMPLETED`, exit `0:0` | `/ari/users/nunal/nanochat-turk/artifacts/tokenizers/bpe_131072/metrics/raw_metrics.json` |

The full TRmorph-reference metric pass for these completed tokenizers completed
as job `494204` (`tok-metrics-bpe-large`), exit `0:0`, elapsed `00:04:34`,
Slurm `CPUTimeRAW=35072`. It computed
`raw_vs_trmorph_reference_metrics.json` for both completed raw BPE tokenizers
and refreshed `docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.*`
with the 64k/128k rows.

A compact tarball containing the completed raw BPE tokenizer bundles was
prepared on UHeM:

```text
/ari/users/nunal/bpe_large_tokenizer_artifacts.tgz
```

Local transfer initially failed because local DNS resolution for
`altay.uhem.itu.edu.tr` intermittently failed during SSH data-transfer commands.
The final transfer used an escalated SSH call to the configured hostname and
mirrored the completed bundles plus full metric files into the local GitHub
checkout.

## 2026-06-22 MorphBPE Completion And Metrics Import

All remaining 64k/128k tokenizer finalizer jobs completed successfully:

| Tokenizer | Job | State | Notes |
| --- | ---: | --- | --- |
| `morphbpe_trmorph_64k` | `494182` | `COMPLETED`, exit `0:0` | Archived under `artifacts/tokenizers/`. |
| `morphbpe_zemberek_64k` | `494183` | `COMPLETED`, exit `0:0` | Archived under `artifacts/tokenizers/`. |
| `morphbpe_tdelight_64k` | `494184` | `COMPLETED`, exit `0:0` | Waited on 32k TurkishDelight finalizer `494159`; archived under `artifacts/tokenizers/`. |
| `morphbpe_trmorph_128k` | `494186` | `COMPLETED`, exit `0:0` | Archived under `artifacts/tokenizers/`. |
| `morphbpe_zemberek_128k` | `494187` | `COMPLETED`, exit `0:0` | Archived under `artifacts/tokenizers/`. |
| `morphbpe_tdelight_128k` | `494188` | `COMPLETED`, exit `0:0` | Waited on 32k TurkishDelight finalizer `494159`; archived under `artifacts/tokenizers/`. |

The TurkishDelight 32k tokenizer also exists and was archived locally as
`artifacts/tokenizers/morphbpe_tdelight_32768/`. The UHeM finalizer job was
`494159`, elapsed `01:35:05`, exit `0:0`. Its `token_bytes.pt` was regenerated
locally from the imported tokenizer using the same byte-accounting logic as
`scripts/tok_train.py`.

The expanded 50k TRmorph-reference tokenizer metric pass completed as job
`496881` (`tok-mbpe-50k`), elapsed `00:18:41`, Slurm `CPUTime=00:59:45`, node
`a071`. It refreshed
`docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md/json`
with 12 local tokenizer rows and wrote the new TurkishDelight metric JSON files.

The first all-tokenizer full-corpus job `494219` failed with exit code `137`
after Slurm killed it for memory (`245.90 GB` used against `192 GB`). The
metrics scheduler was updated to bound in-flight parquet row-group futures, and
the repaired full-corpus job is running as `496882` (`tok-metrics-full`).
