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

# `morphbpe_zemberek_64k`

Exact tokenizer ID in configs and loading snippets: `morphbpe_zemberek_65536`.

This artifact stores a trained nanochat tokenizer bundle for the Turkish
MorphBPE ablation study. It is a raw nanochat/tiktoken tokenizer artifact, not a
Transformers `AutoTokenizer` export.

## Tokenizer Config

```json
{
  "data_dir": "/ari/users/nunal/nanochat-turk-morphbpe-zemberek-32768/segmented/zemberek-segmented",
  "decode_strip": "",
  "doc_cap": 10000,
  "implementation": "morphbpe",
  "iterator_stats": {
    "chars": 2000007497,
    "docs": 737406,
    "docs_with_boundary": 737406
  },
  "max_chars": 2000000000,
  "morph_boundary": "",
  "morph_boundary_codepoints": "U+E000",
  "morphbpe_iterator_stats": {
    "boundary_splits": 132161414,
    "docs": 737406,
    "docs_with_boundary": 737406,
    "regex_chunks": 298657152,
    "training_chunks": 430818566,
    "visible_chars": 1867844008
  },
  "name": "morphbpe_zemberek_65536",
  "requires_runtime_segmentation": false,
  "text_column": "segmented_text",
  "training_boundary_semantics": "merge_constraint_only",
  "training_uses_morph_boundaries": true,
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
- `tokens`: `10506623`
- `tokens_per_byte`: `0.21452857527394212`
- `bytes_per_token`: `4.661383681512128`
- `chars_per_token`: `4.268852322958575`
- `encode_docs_per_sec`: `7010.806456762221`

## Provenance

- Git commit: `unknown`
- Git branch: `unknown`
- Uploaded/generated at: `2026-06-12T15:15:34.423849+00:00`
- Source tokenizer dir: `/ari/users/nunal/nanochat-turk-morphbpe-zemberek-65536/tokenizers/morphbpe_zemberek_65536`

## Loading

Use this repository's `nanochat.tokenizer.RustBPETokenizer.from_directory(...)`
or set:

```bash
export NANOCHAT_BASE_DIR=/path/to/base-dir
export NANOCHAT_TOKENIZER_NAME=morphbpe_zemberek_65536
```
