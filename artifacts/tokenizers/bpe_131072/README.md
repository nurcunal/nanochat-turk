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

# `bpe_128k`

Exact tokenizer ID in configs and loading snippets: `bpe_131072`.

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
  "name": "bpe_131072",
  "requires_runtime_segmentation": false,
  "text_column": "text",
  "training_boundary_semantics": "",
  "training_uses_morph_boundaries": false,
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
- `tokens`: `8613965`
- `tokens_per_byte`: `0.1758835011886886`
- `bytes_per_token`: `5.6855816107913135`
- `chars_per_token`: `5.206803371037612`
- `encode_docs_per_sec`: `7483.541060591057`

## Provenance

- Git commit: `unknown`
- Git branch: `unknown`
- Uploaded/generated at: `2026-06-12T13:57:30.202380+00:00`
- Source tokenizer dir: `/ari/users/nunal/nanochat-turk-bpe-131072/tokenizers/bpe_131072`

## Loading

Use this repository's `nanochat.tokenizer.RustBPETokenizer.from_directory(...)`
or set:

```bash
export NANOCHAT_BASE_DIR=/path/to/base-dir
export NANOCHAT_TOKENIZER_NAME=bpe_131072
```
