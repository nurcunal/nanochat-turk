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
| [`bpe_32k`](bpe_32768/) | none | Raw unsegmented BPE baseline. |
| [`morphbpe_trmorph_32k`](morphbpe_trmorph_32768/) | TRmorph | Archived with raw and TRmorph-reference metrics. |
| [`morphbpe_zemberek_32k`](morphbpe_zemberek_32768/) | Zemberek | Archived with raw 10k and matched TRmorph-reference metrics. |
| [`morphbpe_tdelight_32k`](morphbpe_tdelight_32768/) | TurkishDelightNLP | Archived with raw and 50k TRmorph-reference metrics; tokenizer finalizer job `494159`, 50k metrics job `496881`. |
| [`bpe_64k`](bpe_65536/) | none | Archived with raw and TRmorph-reference metrics; tokenizer job `494181`, metrics job `494204`. |
| [`morphbpe_trmorph_64k`](morphbpe_trmorph_65536/) | TRmorph | Archived with raw and 50k TRmorph-reference metrics; tokenizer job `494182`, 50k metrics jobs `494218`/`496881`. |
| [`morphbpe_zemberek_64k`](morphbpe_zemberek_65536/) | Zemberek | Archived with raw and 50k TRmorph-reference metrics; tokenizer job `494183`, 50k metrics jobs `494218`/`496881`. |
| [`morphbpe_tdelight_64k`](morphbpe_tdelight_65536/) | TurkishDelightNLP | Archived with raw and 50k TRmorph-reference metrics; tokenizer job `494184`, 50k metrics job `496881`. |
| [`bpe_128k`](bpe_131072/) | none | Archived with raw and TRmorph-reference metrics; tokenizer job `494185`, metrics job `494204`. |
| [`morphbpe_trmorph_128k`](morphbpe_trmorph_131072/) | TRmorph | Archived with raw and 50k TRmorph-reference metrics; tokenizer job `494186`, 50k metrics jobs `494218`/`496881`. |
| [`morphbpe_zemberek_128k`](morphbpe_zemberek_131072/) | Zemberek | Archived with raw and 50k TRmorph-reference metrics; tokenizer job `494187`, 50k metrics jobs `494218`/`496881`. |
| [`morphbpe_tdelight_128k`](morphbpe_tdelight_131072/) | TurkishDelightNLP | Archived with raw and 50k TRmorph-reference metrics; tokenizer job `494188`, 50k metrics job `496881`. |

Full-corpus tokenizer metrics are running separately as UHeM job `496882`
(`tok-metrics-full`) and will add `full_trmorph_reference_metrics.json` files
after completion.
