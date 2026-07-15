# Tokenizer Artifacts

Trained tokenizer bundles are archived here after each tokenizer-prep job
finishes. Each tokenizer should live under:

```text
artifacts/tokenizers/<tokenizer_name>/
```

Every bundle has the loadable tokenizer core:

```text
tokenizer.pkl
tokenizer_config.json
token_bytes.pt
README.md
```

Metric JSON files live under `metrics/`. MorphBPE bundles should also carry
segmentation and segmented-dataset manifests under `provenance/`; raw BPE has no
segmentation manifest. A release-ready bundle should have
`provenance/publish_manifest.json`, but the original raw BPE, TRmorph, and
Zemberek 32k bundles predate that schema and do not yet have one. Regenerating
all manifests is tracked in the canonical TODO list.

Large segmented corpora and full production checkpoints are not stored in
these tokenizer bundles; they remain on UHeM or belong on Hugging Face.
Historical smoke model/optimizer checkpoints do exist elsewhere under
`artifacts/` and are retained as provenance. This directory contains the
compact tokenizer outputs needed for loading and evaluation.

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

## Publication Status

All 12 bundles are versioned in this GitHub repository. As of the verified
2026-07-15 release audit, the intended Hugging Face repository
`nurcunal/nanochat-turk-tokenizers` does not exist. The 32k raw BPE tokenizer is
embedded in the public
[`nurcunal/nanochat-turk-d20-bpe32k`](https://huggingface.co/nurcunal/nanochat-turk-d20-bpe32k)
model repository; the other 11 bundles are not on Hugging Face. See the
canonical [publication audit and TODO list](../../MorphBPE-alignment.md#publication-audit).

The 50,000-document comparison is the complete result currently checked in.
Job `496882` (`tok-metrics-full`) was submitted for a full-corpus pass, but no
finished output from that job is present here. Verify its terminal UHeM state
and import its outputs before describing the full-corpus metrics as complete.
