# nanochat-turk Artifacts

This folder contains compact, curated artifacts that make the project easier to
audit from GitHub. It is not intended to mirror every generated file from UHeM.

## Current Artifacts

- [cetvel_base_subset_2026-06-09_job493293](cetvel_base_subset_2026-06-09_job493293/):
  CETVEL base-model subset results for the raw `tr_d20_bpe_32768_chinchilla20`
  model, stopped after `tquad` because the remaining tasks are mostly
  generation/instruction-style diagnostics better suited to a future SFT model.
- [cetvel_core12_model_comparison_2026-06-12](cetvel_core12_model_comparison_2026-06-12/):
  compact common-slice CETVEL comparison for the raw BPE, TRmorph MorphBPE, and
  Zemberek MorphBPE d20 base-model runs.
- [uhem_smoke_2026-06-07_job492393](uhem_smoke_2026-06-07_job492393/):
  completed UHeM A100 smoke test proving the Turkish data, tokenizer,
  pretraining, checkpoint, and BPB-eval path works end to end.
- [tokenizers](tokenizers/):
  compact tokenizer bundles, tokenizer metrics, and tokenizer-prep provenance
  for raw BPE and MorphBPE variants.

## What Belongs Here

Each artifact folder should be self-contained and include:

- `README.md` with a human-readable summary.
- `manifest.json` when provenance matters.
- compact metric summaries such as `metrics.json` or `metrics_summary.json`.
- compressed result archives when they are small enough for GitHub and useful
  for review.
- logs that explain run behavior or failures.

## What Stays Out Of Git

Avoid adding full generated corpora, full-size training caches, large expanded
benchmark folders, and production checkpoints. Prefer UHeM or Hugging Face for
those files, then keep a compact manifest and URL here.

The exception currently retained in git is the tiny d2 smoke-test checkpoint,
because it is an infrastructure artifact rather than a production model.

## Naming Convention

Use descriptive run folders:

```text
<artifact_kind>_<date>_job<slurm_id>/
```

Examples:

```text
uhem_smoke_2026-06-07_job492393/
cetvel_base_subset_2026-06-09_job493293/
cetvel_core12_model_comparison_2026-06-12/
```
