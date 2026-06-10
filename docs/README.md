# nanochat-turk Documentation

This folder keeps the project-facing documentation for the Turkish nanochat
branch. It is organized so the GitHub page answers three questions quickly:
what the project is doing, how the pipeline runs, and how the tokenizer study is
being evaluated.

## Start Here

- [Turkish foundation pipeline](turkish_foundation.md): active training,
  data, A100/UHeM, CETVEL, and run-script workflow.
- [Project report README](project_report_readme.md): running project memory and
  report-writing scaffold.
- [Tokenizer ablation plan](tokenizer_ablation_plan.md): controlled MorphBPE
  and tokenizer comparison plan.

## Tokenizer Study Notes

Tokenizer-specific implementation notes and small reproducible result summaries
live under [tokenizer_tests/](tokenizer_tests/):

- MorphBPE framework and raw-text inference contract.
- Turkish segmenter benchmarks and examples.
- Local/LLM judge workflow and committed small judge outputs.
- TurkishDelightNLP setup notes.

## Organization Rules

- Keep top-level files in `docs/` for stable project plans, pipeline docs, and
  report-writing scaffolds.
- Keep tokenizer and morphology experiment notes under `docs/tokenizer_tests/`.
- Keep small, reviewable result summaries in git when they support the report.
- Keep large generated corpora, full model checkpoints, and large expanded
  benchmark outputs on UHeM or Hugging Face, with only compact manifests or
  archives under `artifacts/` when useful.
