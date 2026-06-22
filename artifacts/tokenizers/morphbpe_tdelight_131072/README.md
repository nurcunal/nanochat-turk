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

# `morphbpe_tdelight_131072`

This artifact stores a trained nanochat tokenizer bundle for the Turkish
MorphBPE ablation study. It is a raw nanochat/tiktoken tokenizer artifact, not a
Transformers `AutoTokenizer` export.

## Tokenizer Config

```json
{
  "data_dir": "/ari/users/nunal/nanochat-turk-morphbpe-tdelight-32768/segmented/tdelight-segmented",
  "decode_strip": "",
  "doc_cap": 10000,
  "implementation": "morphbpe",
  "iterator_stats": {
    "chars": 2000001793,
    "docs": 735229,
    "docs_with_boundary": 735229
  },
  "max_chars": 2000000000,
  "morph_boundary": "",
  "morph_boundary_codepoints": "U+E000",
  "morphbpe_iterator_stats": {
    "boundary_splits": 139333916,
    "docs": 735229,
    "docs_with_boundary": 735229,
    "regex_chunks": 297521010,
    "training_chunks": 436854926,
    "visible_chars": 1860663994
  },
  "name": "morphbpe_tdelight_131072",
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
- `tokens`: `9329615`
- `tokens_per_byte`: `0.1904959389714849`
- `bytes_per_token`: `5.24945573852726`
- `chars_per_token`: `4.807403306567313`
- `encode_docs_per_sec`: `7028.40970882474`

## Provenance

- Git commit: `unknown`
- Git branch: `unknown`
- Uploaded/generated at: `2026-06-12T18:49:07.194833+00:00`
- Source tokenizer dir: `/ari/users/nunal/nanochat-turk-morphbpe-tdelight-131072/tokenizers/morphbpe_tdelight_131072`

## Loading

Use this repository's `nanochat.tokenizer.RustBPETokenizer.from_directory(...)`
or set:

```bash
export NANOCHAT_BASE_DIR=/path/to/base-dir
export NANOCHAT_TOKENIZER_NAME=morphbpe_tdelight_131072
```
