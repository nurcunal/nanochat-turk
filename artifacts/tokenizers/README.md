# Tokenizer Artifacts

Trained tokenizer bundles are archived here after each tokenizer-prep job
finishes. Each tokenizer should live under:

```text
artifacts/tokenizers/<tokenizer_name>/
```

The expected bundle files are:

```text
tokenizer.pkl
tokenizer_config.json
token_bytes.pt
metrics/raw_metrics.json
provenance/publish_manifest.json
provenance/segmentation_manifest.json
provenance/segmented_dataset_manifest.json
README.md
```

Large segmented corpora and model checkpoints are not stored in GitHub. They
remain on UHeM/Hugging Face; this directory is for the compact tokenizer outputs
needed to reproduce tokenizer loading and evaluation.

## Current Bundles

| Tokenizer | Segmenter | Status |
| --- | --- | --- |
| [`bpe_32768`](bpe_32768/) | none | Raw unsegmented BPE baseline. |
| [`morphbpe_trmorph_32768`](morphbpe_trmorph_32768/) | TRmorph | Archived with raw and TRmorph-reference metrics. |
| [`morphbpe_zemberek_32768`](morphbpe_zemberek_32768/) | Zemberek | Archived with raw 10k and matched TRmorph-reference metrics. |
| [`bpe_65536`](bpe_65536/) | none | Archived with raw and TRmorph-reference metrics; tokenizer job `494181`, metrics job `494204`. |
| `morphbpe_trmorph_65536` | TRmorph | UHeM tokenizer job `494182`; archive after completion. |
| `morphbpe_zemberek_65536` | Zemberek | UHeM tokenizer job `494183`; archive after completion. |
| `morphbpe_tdelight_65536` | TurkishDelightNLP | UHeM tokenizer job `494184`; waits on TurkishDelight 32k finalizer `494159`. |
| [`bpe_131072`](bpe_131072/) | none | Archived with raw and TRmorph-reference metrics; tokenizer job `494185`, metrics job `494204`. |
| `morphbpe_trmorph_131072` | TRmorph | UHeM tokenizer job `494186`; archive after completion. |
| `morphbpe_zemberek_131072` | Zemberek | UHeM tokenizer job `494187`; archive after completion. |
| `morphbpe_tdelight_131072` | TurkishDelightNLP | UHeM tokenizer job `494188`; waits on TurkishDelight 32k finalizer `494159`. |
