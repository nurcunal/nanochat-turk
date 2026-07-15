---
language:
- tr
tags:
- nanochat
- turkish
- tokenizer
- bpe
library_name: tiktoken
---

# `bpe_64k`

Exact tokenizer ID in configs and loading snippets: `bpe_65536`.

This artifact stores a trained nanochat tokenizer bundle for the Turkish
MorphBPE ablation study. It is a raw nanochat/tiktoken tokenizer artifact, not a
Transformers `AutoTokenizer` export.

## Tokenizer Config

```json
{
  "data_dir": "/ari/users/nunal/nanochat-turk-d20-bpe32k/base_data_fineweb2_tur_latn",
  "decode_strip": "",
  "doc_cap": 10000,
  "implementation": "bpe",
  "iterator_stats": {
    "chars": 2000001023,
    "docs": 777394,
    "docs_with_boundary": 0
  },
  "max_chars": 2000000000,
  "morph_boundary": "",
  "morph_boundary_codepoints": "",
  "morphbpe_iterator_stats": {},
  "name": "bpe_65536",
  "requires_runtime_segmentation": false,
  "text_column": "text",
  "training_boundary_semantics": "",
  "training_uses_morph_boundaries": false,
  "vocab_size": 65536
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
- `tokens`: `9212043`
- `tokens_per_byte`: `0.18809530523292703`
- `bytes_per_token`: `5.316453798576494`
- `chars_per_token`: `4.868759514040479`
- `encode_docs_per_sec`: `6723.93729332886`

## Provenance

- Git commit: `unknown`
- Git branch: `unknown`
- Uploaded/generated at: `2026-06-12T13:57:26.265154+00:00`
- Source tokenizer dir: `/ari/users/nunal/nanochat-turk-bpe-65536/tokenizers/bpe_65536`

## Loading

Use this repository's `nanochat.tokenizer.RustBPETokenizer.from_directory(...)`
or set:

```bash
export NANOCHAT_BASE_DIR=/path/to/base-dir
export NANOCHAT_TOKENIZER_NAME=bpe_65536
```
