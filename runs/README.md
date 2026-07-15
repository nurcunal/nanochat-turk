# Run Launcher Index

The scripts in this directory include reusable launchers and dated UHeM/Slurm
operations retained for reproducibility. They are not all active recommendations.
Before reusing an old launcher, compare its tokenizer name, model tag, data
revision, cache paths, dependency job IDs, and environment setup with the
current study status in [`MorphBPE-alignment.md`](../MorphBPE-alignment.md).

## Local and Upstream Workflows

| Launcher | Purpose |
| --- | --- |
| `runcpu.sh` | Small CPU development run. |
| `miniseries.sh` | Upstream nanochat model-series workflow. |
| `scaling_laws.sh` | Upstream scaling-law workflow. |
| `speedrun.sh` | Upstream time-to-GPT-2 workflow. |
| `turkish_smoke.sh` | Small Turkish end-to-end smoke test. |
| `turkish_foundation.sh` | Turkish foundation-model pipeline. |
| `turkish_ablation.sh` | Turkish tokenizer/model ablation wrapper. |

## UHeM Training

| Launcher family | Purpose |
| --- | --- |
| `uhem_a100.sbatch`, `uhem_smoke_*.sbatch` | Environment and GPU smoke checks. |
| `uhem_nakane_prepare_*.sbatch` | Prepare raw or morphology-constrained 32k tokenizers. |
| `uhem_nakane_segment_*.sbatch` | Materialize boundary-marked corpora by segmenter. |
| `uhem_nakane_finalize_*.sbatch` | Finalize and archive tokenizer bundles. |
| `uhem_nakane_a100x4_*.sbatch` | Full base-model training jobs. |
| `uhem_submit_64k_128k_tokenizer_ablation.sh` | Historical 64k/128k tokenizer and model grid submission. |

## Metrics, Evaluation, and Publication

| Launcher family | Purpose |
| --- | --- |
| `uhem_tokenizer_metrics_*.sbatch` | Sample/full tokenizer metrics and external references. |
| `uhem_cetvel_full_final.sbatch` | CETVEL evaluation for a selected checkpoint. |
| `uhem_submit_cetvel_full_final.sh` | Submit CETVEL plus its dependent Hub upload. |
| `uhem_publish_morphbpe_trmorph_32k_tokenizer.sbatch` | Historical consolidated-tokenizer Hub attempt. |
| `uhem_upload_final_to_hf.sbatch` | Upload a raw nanochat checkpoint bundle to Hugging Face. |

## Safety Notes

- The numbered Slurm job IDs in comments and docs are historical provenance,
  not reusable dependencies.
- Keep every tokenizer in a distinct `NANOCHAT_TOKENIZER_NAME` and every model
  in a matching model tag.
- Production checkpoints should go to Hugging Face or controlled storage, not
  into Git history.
- Review local `git diff` before submitting jobs so uncommitted cluster-specific
  changes are not mistaken for the published launcher configuration.
