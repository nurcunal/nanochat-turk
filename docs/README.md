# Documentation Guide

The canonical project assessment is the root
[`MorphBPE-alignment.md`](../MorphBPE-alignment.md). It records the current
paper comparison, completed evidence, publication state, limitations, and TODO
list. Use this index to distinguish current result sources from historical
operations notes.

## Current Research Sources

| Document | Role |
| --- | --- |
| [`MorphBPE-alignment.md`](../MorphBPE-alignment.md) | Canonical status, claims, release audit, and priorities. |
| [`tokenizer_ablation_plan.md`](tokenizer_ablation_plan.md) | Controlled study design and matched model matrix. |
| [`model_bpb_inventory.md`](model_bpb_inventory.md) | Completed model tags, steps, and validation BPB. |
| [`cetvel_model_comparison.md`](cetvel_model_comparison.md) | Current common-slice CETVEL model results. |
| [`tokenizer_tests/tokenizer_metrics/`](tokenizer_tests/tokenizer_metrics/) | Current 12-tokenizer intrinsic metrics and external references. |
| [`tokenizer_tests/morphbpe_framework.md`](tokenizer_tests/morphbpe_framework.md) | Method implementation and raw-text inference contract. |

## Pipeline and Reproduction Guides

| Document | Role |
| --- | --- |
| [`turkish_foundation.md`](turkish_foundation.md) | Data, training-horizon, UHeM, and CETVEL pipeline reference. |
| [`tokenizer_tests/README.md`](tokenizer_tests/README.md) | Tokenizer/morphology documentation index. |
| [`tokenizer_tests/llm_judge_pipeline.md`](tokenizer_tests/llm_judge_pipeline.md) | Blind segmenter-judging workflow. |
| [`tokenizer_tests/turkishdelight_setup.md`](tokenizer_tests/turkishdelight_setup.md) | TurkishDelightNLP environment notes. |
| [`../runs/README.md`](../runs/README.md) | Local and Slurm launcher index. |

## Historical Records

These files remain useful for provenance, but point-in-time plans and job states
inside them are not current status:

- [`project_report_readme.md`](project_report_readme.md) - historical long-form
  project memory and report scaffold.
- [`tokenizer_tests/vocab64_128_uhem_launch.md`](tokenizer_tests/vocab64_128_uhem_launch.md)
  - 64k/128k launch record.
- [`tokenizer_tests/uhem_restart_notes.md`](tokenizer_tests/uhem_restart_notes.md)
  - exact pause/restart operations.
- [`tokenizer_tests/morphbpe_trmorph_32k_uhem_operations.md`](tokenizer_tests/morphbpe_trmorph_32k_uhem_operations.md)
  - 32k TRmorph job record.
- [`tokenizer_tests/segmenter_benchmark_status.md`](tokenizer_tests/segmenter_benchmark_status.md)
  - detailed initial segmenter benchmark log.

## Organization Rules

- Keep one current claim/status document: `MorphBPE-alignment.md`.
- Keep stable design and result summaries in `docs/`.
- Keep tokenizer-specific material under `docs/tokenizer_tests/`.
- Label dated launch, restart, and cluster notes as historical records.
- Keep compact reviewable outputs in `artifacts/`; publish large weights and
  distributable result bundles on Hugging Face with manifests and checksums.
- Never infer completion from a submitted Slurm job ID. Import its terminal
  state and output before updating current documentation.
