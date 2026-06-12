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
| [`morphbpe_zemberek_32768`](morphbpe_zemberek_32768/) | Zemberek | Archived with raw 10k tokenizer metric; matched boundary-reference metric pending. |
