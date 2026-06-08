# MorphBPE Boundary Framework

MorphBPE tokenizer variants must preserve the original Turkish surface text
while preventing BPE training from treating morpheme boundaries as normal
mergeable text.

## Boundary Marker

The internal morpheme boundary marker is:

```text
U+E000
```

It is defined in:

```text
nanochat/morphology/boundary.py
```

Segmented corpus text should look like this internally:

```text
evlerden çalışıyorum
```

Tokenizer decoding strips the marker, so the visible decoded text is:

```text
evlerden çalışıyorum
```

This gives the tokenizer an internal morpheme-boundary signal without training
the model to emit whitespace-separated Turkish morphemes.

## Corpus Segmentation

`scripts/morph_segment_corpus.py` now uses the internal boundary marker as its
default delimiter. It writes `segmented_text` plus per-row segmentation stats and
a manifest recording the delimiter codepoint. It also writes a
`fineweb2_manifest.json` compatibility manifest so `nanochat.dataset` preserves
the segmented shard order and still uses the final shard as validation.

Example:

```bash
TRMORPH_SEGMENT_FST=/private/tmp/TRmorph/segment.fst \
TRMORPH_FLOOKUP_FLAGS=-x \
TRMORPH_MAX_OUTPUT_LINES_PER_WORD=512 \
TRMORPH_MAX_ANALYSES_PER_WORD=64 \
python3 -m scripts.morph_segment_corpus \
  --backend trmorph \
  --data-dir /path/to/base_data_fineweb2_tur_latn \
  --output-dir /path/to/segmented/trmorph \
  --compact \
  --segment-batch-size 2048
```

Repeat the same corpus materialization for:

```text
trmorph
tdelight
zemberek
```

## Tokenizer Training

`scripts/tok_train.py` supports:

```text
--implementation morphbpe
--data-dir /path/to/segmented/<backend>
--text-column segmented_text
```

Example:

```bash
NANOCHAT_TOKENIZER_NAME=morphbpe_trmorph_32768 \
python3 scripts/tok_train.py \
  --implementation morphbpe \
  --data-dir /path/to/segmented/trmorph \
  --text-column segmented_text \
  --vocab-size 32768
```

The saved tokenizer config includes:

```text
implementation: morphbpe
morph_boundary: U+E000 marker
decode_strip: U+E000 marker
text_column: segmented_text
```

`RustBPETokenizer.from_directory()` reads `tokenizer_config.json` and strips the
boundary marker on decode.

Tokenizer metrics can be run on the same internal segmented text while reporting
visible bytes/words after boundary stripping:

```bash
python3 -m scripts.tokenizer_metrics \
  --tokenizer-dir /path/to/tokenizers/morphbpe_trmorph_32768 \
  --data-dir /path/to/segmented/trmorph \
  --text-column segmented_text \
  --max-docs 10000
```

## Pretraining Use

For full model training with a MorphBPE tokenizer, point the normal nanochat
dataloader at the segmented shard directory before launching `base_train`:

```bash
export NANOCHAT_TOKENIZER_NAME=morphbpe_trmorph_32768
export NANOCHAT_DATA_DIR=/path/to/segmented/trmorph
export NANOCHAT_TEXT_COLUMN=segmented_text

torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
  --depth=20 \
  --model-tag=tr_morphbpe_trmorph_32768_d20
```

The dataloader reads `NANOCHAT_TEXT_COLUMN`, so the model is trained on the
internal segmented stream. Validation BPB uses the tokenizer's `token_bytes.pt`,
which was generated with boundary stripping, so bytes are counted over the
visible Turkish surface text.

## Reversibility Contract

For a segmented document:

```python
decoded = tokenizer.decode(tokenizer.encode(segmented_text))
assert decoded == segmented_text.replace(MORPHEME_BOUNDARY, "")
```

For tokenizer evaluation and BPB, `token_bytes.pt` is computed after decode-time
boundary stripping. Boundary-only tokens therefore count as zero bytes, and
tokens containing a boundary plus surface text count only the surface bytes.

## Current Scope

This framework prepares MorphBPE training and pretraining from pre-segmented
corpora. It does not yet make inference-time raw prompts morphology-aware unless
they are pre-segmented before encoding. For controlled pretraining ablations,
use segmented train and validation corpora with `NANOCHAT_DATA_DIR` and
`NANOCHAT_TEXT_COLUMN=segmented_text`.

Before running full UHeM training, run tokenizer-only checks for each tokenizer:

- encode/decode reversibility;
- BPB/compression;
- tokens per byte/word;
- boundary-marker token frequency;
- vocab entries containing `U+E000`;
- throughput.
