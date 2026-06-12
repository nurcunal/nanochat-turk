# Tokenizer And Morphology Test Notes

This folder contains the tokenizer-study documentation for `nanochat-turk`.
It is intentionally narrower than the main `docs/` folder: everything here
supports the Turkish morphology-aware tokenizer ablation.

## Files

- [MorphBPE framework](morphbpe_framework.md): explains `morphbpe` versus
  `preseg_bpe`, the `U+E000` boundary marker, and raw-text inference.
- [Segmenter benchmark status](segmenter_benchmark_status.md): current
  TRmorph, Zemberek, TurkishDelightNLP, and identity benchmark results.
- [Segmenter examples](segmenter_examples.md): small original-vs-segmented
  examples for quick inspection.
- [LLM judge pipeline](llm_judge_pipeline.md): blind segmenter-quality pack
  generation and scoring workflow.
- [Codex-local judge results](codex_local_judge_results.md): no-API local
  judge methodology and aggregate scores.
- [TurkishDelightNLP setup](turkishdelight_setup.md): runtime setup notes for
  the TurkishDelight wrapper.
- [UHeM restart notes](uhem_restart_notes.md): intentional pauses/cancellations
  and exact restart commands for cluster jobs.
- [Tokenizer metric ranking methodology](tokenizer_metrics/ranking_methodology.md):
  diagnostic score formula, weight rationale, and sensitivity caveats.

## Small Result Files

The [judge_results/](judge_results/) folder contains compact committed judge
outputs:

- `codex_local_hash_100k_judge_500.judgments.jsonl`
- `codex_local_hash_100k_judge_500.scores.json`

Large generated packs, full segmented corpora, and benchmark caches remain
outside git under `dev-ignore/` or on UHeM/Hugging Face.

## Folder Policy

Add new files here when they are specifically about tokenizer behavior,
morphological segmentation, MorphBPE, SentencePiece controls, tokenizer
metrics, or segmenter judging. Put broader project plans and report scaffolds
in the parent `docs/` folder.
