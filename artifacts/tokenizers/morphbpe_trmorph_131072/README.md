---
language:
- tr
tags:
- nanochat
- turkish
- tokenizer
- morphbpe
library_name: tiktoken
---

# `morphbpe_trmorph_131072`

This artifact stores a trained nanochat tokenizer bundle for the Turkish
MorphBPE ablation study. It is a raw nanochat/tiktoken tokenizer artifact, not a
Transformers `AutoTokenizer` export.

## Tokenizer Config

```json
{
  "data_dir": "/ari/users/nunal/nanochat-turk-morphbpe-trmorph-32768/segmented/trmorph",
  "decode_strip": "",
  "doc_cap": 10000,
  "implementation": "morphbpe",
  "iterator_stats": {
    "chars": 2000003945,
    "docs": 734628,
    "docs_with_boundary": 734624
  },
  "max_chars": 2000000000,
  "morph_boundary": "",
  "morph_boundary_codepoints": "U+E000",
  "morphbpe_iterator_stats": {
    "boundary_splits": 141376276,
    "docs": 734628,
    "docs_with_boundary": 734624,
    "regex_chunks": 297199201,
    "training_chunks": 438575477,
    "visible_chars": 1858612668
  },
  "name": "morphbpe_trmorph_131072",
  "requires_runtime_segmentation": false,
  "text_column": "segmented_text",
  "training_boundary_semantics": "merge_constraint_only",
  "training_uses_morph_boundaries": true,
  "vocab_size": 131072
}
```

## Included Files

- `tokenizer.pkl`
- `tokenizer_config.json`
- `token_bytes.pt`
- `metrics/raw_metrics.json` when available
- `provenance/segmentation_manifest.json` when available
- `provenance/segmented_dataset_manifest.json` when available
- `provenance/publish_manifest.json`

## Metrics

- `docs`: `10000`
- `bytes`: `48975401`
- `tokens`: `10047209`
- `tokens_per_byte`: `0.20514807015056397`
- `bytes_per_token`: `4.874527941043129`
- `chars_per_token`: `4.464047876380396`
- `encode_docs_per_sec`: `6691.225972199445`

## Provenance

- Git commit: `unknown`
- Git branch: `unknown`
- Uploaded/generated at: `2026-06-12T15:18:40.570084+00:00`
- Source tokenizer dir: `/ari/users/nunal/nanochat-turk-morphbpe-trmorph-131072/tokenizers/morphbpe_trmorph_131072`

## Loading

Use this repository's `nanochat.tokenizer.RustBPETokenizer.from_directory(...)`
or set:

```bash
export NANOCHAT_BASE_DIR=/path/to/base-dir
export NANOCHAT_TOKENIZER_NAME=morphbpe_trmorph_131072
```
