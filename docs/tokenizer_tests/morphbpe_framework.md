# MorphBPE Framework

This repository supports two different morphology-aware tokenizer experiments:

```text
morphbpe    = main method; boundary-constrained BPE training, raw-text inference
preseg_bpe  = control; BPE trained and modeled on boundary-marked text
```

The main paper method should be `morphbpe`.

## Main Method

Paper-faithful MorphBPE uses a morphological segmenter only while learning BPE
merges. The segmenter marks surface morpheme boundaries:

```text
ev<U+E000>ler<U+E000>den
```

During tokenizer training, this becomes independent BPE training chunks:

```text
ev
ler
den
```

Pairs inside a morpheme can be merged. Pairs crossing morpheme boundaries are
never counted by the BPE trainer, so merges such as `ev + ler -> evler` are not
learned from that segmented occurrence.

The saved tokenizer is still a normal raw-text BPE tokenizer:

```text
input:  evlerden geldik
decode: evlerden geldik
```

No runtime segmenter is required for normal user prompts, CETVEL prompts, or
chat usage.

## Boundary Marker

The internal boundary marker is:

```text
U+E000
```

It is defined in:

```text
nanochat/morphology/boundary.py
```

For `morphbpe`, this marker is a tokenizer-training annotation only and is not
saved as a decode-strip rule. For `preseg_bpe`, the marker is part of the
internal model text stream and is stripped on decode.

## Corpus Segmentation

`scripts/morph_segment_corpus.py` uses `U+E000` as its default delimiter. It
writes `segmented_text` plus per-row segmentation stats and a full segmentation
manifest. It also writes `fineweb2_manifest.json` so `nanochat.dataset` can
preserve shard order and the final-shard validation convention.

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

Repeat corpus materialization for:

```text
trmorph
tdelight
zemberek
```

## MorphBPE Tokenizer Training

Train the main MorphBPE tokenizer from the segmented annotation column:

```bash
NANOCHAT_TOKENIZER_NAME=morphbpe_trmorph_32768 \
python3 scripts/tok_train.py \
  --implementation morphbpe \
  --data-dir /path/to/segmented/trmorph \
  --text-column segmented_text \
  --vocab-size 32768
```

The saved config records:

```text
implementation: morphbpe
training_boundary_semantics: merge_constraint_only
requires_runtime_segmentation: false
decode_strip: ""
```

Then evaluate tokenizer metrics on raw Turkish text, not on `segmented_text`:

```bash
python3 -m scripts.tokenizer_metrics \
  --tokenizer-dir /path/to/tokenizers/morphbpe_trmorph_32768 \
  --data-dir /path/to/base_data_fineweb2_tur_latn \
  --text-column text \
  --max-docs 10000
```

## Pretraining Use

For full model training with the main MorphBPE tokenizer, keep the normal raw
FineWeb-2 Turkish data stream:

```bash
export NANOCHAT_TOKENIZER_NAME=morphbpe_trmorph_32768
export NANOCHAT_DATA_DIR=/path/to/base_data_fineweb2_tur_latn
export NANOCHAT_TEXT_COLUMN=text

torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
  --depth=20 \
  --model-tag=tr_morphbpe_trmorph_32768_d20
```

The morphological segmenter is not used during LLM pretraining or inference.
Its only role is to constrain tokenizer merge learning.

## Pre-Segmented BPE Control

`preseg_bpe` is a useful ablation, but it is not the main MorphBPE method.
Here, the model trains on boundary-marked text:

```bash
NANOCHAT_TOKENIZER_NAME=preseg_bpe_trmorph_32768 \
python3 scripts/tok_train.py \
  --implementation preseg_bpe \
  --data-dir /path/to/segmented/trmorph \
  --text-column segmented_text \
  --vocab-size 32768
```

For this control, the saved config records:

```text
implementation: preseg_bpe
training_boundary_semantics: visible_presegmented_control
requires_runtime_segmentation: true
decode_strip: U+E000 marker
```

Use this control to test whether gains come from the merge table itself or from
explicitly showing morpheme boundaries to the model.

## Reversibility Contracts

Main MorphBPE:

```python
decoded = tokenizer.decode(tokenizer.encode("evlerden geldik"))
assert decoded == "evlerden geldik"
```

Pre-segmented BPE control:

```python
decoded = tokenizer.decode(tokenizer.encode(segmented_text))
assert decoded == segmented_text.replace(MORPHEME_BOUNDARY, "")
```

Before full UHeM training, run tokenizer-only checks:

- encode/decode reversibility;
- BPB/compression on raw validation text;
- tokens per byte/word;
- percentage of candidate merge tokens that would cross known boundaries;
- throughput.
