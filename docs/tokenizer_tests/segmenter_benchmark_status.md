# Turkish Segmenter Benchmark Status

This document tracks the first morphology-segmenter checks for the Turkish
MorphBPE tokenizer study. Large data and JSON outputs live under
`dev-ignore/` and are intentionally not committed.

Small original-vs-segmented examples are committed in
`docs/tokenizer_tests/segmenter_examples.md`.

The blind LLM/human judge workflow is documented in
`docs/tokenizer_tests/llm_judge_pipeline.md`.

TurkishDelightNLP runtime setup notes are documented in
`docs/tokenizer_tests/turkishdelight_setup.md`.

MorphBPE boundary-marker and tokenizer-training framework notes are documented
in `docs/tokenizer_tests/morphbpe_framework.md`.

## Corpus Cache

Permanent local cache:

```text
dev-ignore/morph-smoke/base_data_fineweb2_tur_latn/
```

Current files:

- `000_00000.parquet`: first FineWeb-2 Turkish training shard, about 4.5 GiB.
- `004_00005.parquet`: validation shard downloaded by the nanochat dataset
  helper, about 2.6 GiB.
- `fineweb2_manifest.json`: shard manifest.

Full first-shard inventory from `000_00000.parquet`:

| Field | Value |
| --- | ---: |
| Documents | 3,381,000 |
| Word-like tokens | 1,282,426,655 |
| Unique word forms | 10,829,544 |
| Characters | 10,026,589,625 |
| UTF-8 bytes | 10,981,374,221 |

Reusable word-count cache:

```text
dev-ignore/morph-smoke/benchmarks/word_counts_000_00000.pkl
```

Summary JSON:

```text
dev-ignore/morph-smoke/benchmarks/word_counts_000_00000_summary.json
```

## Benchmark Metrics

Each segmenter benchmark reports:

- `unique_words`: unique word types processed by that backend.
- `weighted_words`: corpus-token count represented by the processed types.
- `elapsed_seconds`, `unique_words_per_sec`, `weighted_words_per_sec`.
- `pieces_per_word_types`: mean morpheme pieces per unique type.
- `pieces_per_word_weighted`: mean morpheme pieces per corpus token.
- `split_word_rate_types`: fraction of processed unique types split into
  multiple pieces.
- `split_word_rate_weighted`: frequency-weighted split rate.
- `fallback_rate_types`: fraction of processed unique types preserved as
  identity fallback because the segmenter gave no usable surface segmentation.
- `fallback_rate_weighted`: frequency-weighted fallback rate.
- `fallback_reasons_*`: top reasons for fallback, such as no analysis, invalid
  surface reconstruction, timeout, or output overflow.

These are segmenter-readiness metrics, not final tokenizer metrics. Final
tokenizer evaluation must also measure boundary violations, BPB/compression,
token fertility, encode throughput, and downstream CETVEL/core benchmark
results after model training.

## Full Segmented Corpus Artifacts

Full segmented corpus files should be generated locally, not committed to git.
Use this permanent layout:

```text
dev-ignore/morph-smoke/segmented/
  trmorph/
    000_00000.segmented.parquet
    manifest.json
  zemberek/
    000_00000.segmented.parquet
    manifest.json
  tdelight/
    000_00000.segmented.parquet
    manifest.json
```

Preferred parquet columns:

- `id`: source document id from FineWeb-2 when available.
- `text`: original document text. This can be omitted in a compact variant
  because the raw shard is already cached locally.
- `segmented_text`: surface-morpheme segmented text.
- `segmenter`: backend name and version/config.
- `fallback_words`: number of word-like tokens that used identity fallback.
- `split_words`: number of word-like tokens split into multiple pieces.

Each `manifest.json` should include source shard path, command, environment
variables, git commit, segmenter version/config, row counts, byte counts, start
and end timestamps, and output checksum.

Git should only contain the scripts, docs, small examples, and aggregate
metrics. The full original and segmented corpora remain available locally under
`dev-ignore/morph-smoke/`.

Materialization script:

```text
scripts/morph_segment_corpus.py
```

Example TRmorph materialization command:

```bash
TRMORPH_SEGMENT_FST=/private/tmp/TRmorph/segment.fst \
TRMORPH_FLOOKUP_FLAGS=-x \
TRMORPH_MAX_OUTPUT_LINES_PER_WORD=512 \
TRMORPH_MAX_ANALYSES_PER_WORD=64 \
NANOCHAT_BASE_DIR=/Users/nurcunal/Documents/nanochat-turk/dev-ignore/morph-smoke \
python3 -m scripts.morph_segment_corpus \
  --backend trmorph \
  --max-files 1 \
  --output-dir dev-ignore/morph-smoke/segmented/trmorph \
  --segment-batch-size 2048
```

A small identity proof-of-write has already been created locally:

```text
dev-ignore/morph-smoke/segmented/identity_smoke/000_00000.segmented.parquet
dev-ignore/morph-smoke/segmented/identity_smoke/manifest.json
```

## Measurement Plan

Measure segmentation in four layers:

1. Systems performance: elapsed seconds, unique word types/sec, weighted corpus
   words/sec, batch size, timeout/cap settings, and cache reuse.
2. Coverage and robustness: type-weighted and frequency-weighted fallback rates,
   fallback reasons, timeout/overflow counts, and exact reconstruction failures.
3. Segmentation behavior: pieces/word, split rate, piece-count distribution,
   sample segmentations, and cross-segmenter agreement/disagreement.
4. Gold/manual quality: boundary precision, recall, F1, exact-word segmentation
   accuracy, and morpheme edit distance on a Turkish gold or hand-annotated
   evaluation set.

Tokenizer-level metrics are separate: boundary violation rate after tokenizer
training, BPB/compression, tokens per byte/char/word, encode throughput, and
model benchmark results.

## Current Results

### Identity Baseline

Full first-shard result:

```text
dev-ignore/morph-smoke/benchmarks/identity_full_shard_cached.json
```

| Metric | Value |
| --- | ---: |
| Processed unique word forms | 10,829,544 |
| Processed weighted words | 1,282,426,655 |
| Runtime | 35.03 s |
| Unique forms / second | 309,135 |
| Pieces / word, weighted | 1.000 |
| Split rate, weighted | 0.000 |
| Fallback rate, weighted | 0.000 |

### TRmorph

Backend setup:

- Source checkout/build used for this local test: `/private/tmp/TRmorph`.
- FST used: `/private/tmp/TRmorph/segment.fst`.
- Runtime command: `flookup -x /private/tmp/TRmorph/segment.fst`.
- Output controls used for full-shard run:
  - `TRMORPH_MAX_OUTPUT_LINES_PER_WORD=512`
  - `TRMORPH_MAX_ANALYSES_PER_WORD=64`
  - `--batch-size 2048`
  - `--timeout 60`

Full first-shard result target:

```text
dev-ignore/morph-smoke/benchmarks/trmorph_full_shard.json
```

Full first-shard feasibility result:

| Metric | Value |
| --- | ---: |
| Processed unique word forms | 10,829,544 |
| Processed weighted words | 1,282,426,655 |
| Runtime | 1,286.69 s |
| Unique forms / second | 8,417 |
| Weighted words / second | 996,687 |
| Pieces / word, weighted | 1.474 |
| Split rate, weighted | 0.292 |
| Fallback rate, weighted | 0.390 |
| Fallback rate, types | 0.874 |

This full run proved that TRmorph can process the entire first shard when
`flookup` output is streamed and capped. However, it was run before the parser
added case-only surface repair and generic analysis-tag dropping, so its
fallback rates should be treated as a conservative pre-repair estimate. The
current code should be rerun for final full-shard TRmorph numbers.

Corrected hash-100k screening result:

```text
dev-ignore/morph-smoke/benchmarks/trmorph_hash_100k.json
```

| Metric | Value |
| --- | ---: |
| Processed unique word forms | 100,000 |
| Processed weighted words | 12,095,210 |
| Runtime | 10.84 s |
| Unique forms / second | 9,223 |
| Pieces / word, weighted | 1.542 |
| Split rate, weighted | 0.338 |
| Fallback rate, weighted | 0.145 |
| Fallback rate, types | 0.817 |
| Fallback reasons, types | `parse_failed`: 79,106; `analysis_output_overflow:512`: 2,551 |

Smoke examples:

| Word | Pieces |
| --- | --- |
| `evlerden` | `ev + ler + den` |
| `geldik` | `gel + dik` |
| `kitaplarım` | `kitap + lar + ım` |
| `çalışıyorum` | `çalış + ıyor + um` |

### Zemberek

Repo wrapper:

```text
scripts/zemberek_segment_cmd.py
```

Local smoke environment:

```text
/private/tmp/zemberek-smoke-py312
```

Reason: `zemberek-python` imported successfully under Python 3.12 in this
environment, while the system Python 3.13 path was not compatible with its ANTLR
dependency.

Smoke examples:

| Word | Pieces |
| --- | --- |
| `evlerden` | `ev + ler + den` |
| `geldik` | `gel + di + k` |
| `kitaplarım` | `kitap + lar + ım` |
| `çalışıyorum` | `çalış + ıyor + um` |

No full-shard run has been attempted yet. Use the bounded protocol below first.

Corrected hash-1k smoke result:

```text
dev-ignore/morph-smoke/benchmarks/zemberek_hash_1k.json
```

| Metric | Value |
| --- | ---: |
| Processed unique word forms | 1,000 |
| Processed weighted words | 76,484 |
| Runtime | 2.00 s |
| Unique forms / second | 500 |
| Pieces / word, weighted | 1.468 |
| Split rate, weighted | 0.247 |
| Fallback rate, weighted | 0.426 |
| Fallback rate, types | 0.250 |
| Fallback reasons, types | `parse_failed`: 250 |

This uses the command wrapper, which is suitable for smoke tests. For large
Zemberek corpus materialization, prefer running `scripts.morph_segment_corpus`
inside a Python 3.12 environment that has both `zemberek-python` and `pyarrow`,
so the Zemberek model can be loaded once.

Corrected hash-100k screening result:

```text
dev-ignore/morph-smoke/benchmarks/zemberek_hash_100k.json
```

| Metric | Value |
| --- | ---: |
| Processed unique word forms | 100,000 |
| Processed weighted words | 12,095,210 |
| Runtime | 29.99 s |
| Unique forms / second | 3,335 |
| Pieces / word, weighted | 1.447 |
| Split rate, weighted | 0.316 |
| Fallback rate, weighted | 0.425 |
| Fallback rate, types | 0.233 |
| Fallback reasons, types | `parse_failed`: 23,344 |

### TurkishDelightNLP

Repo wrapper:

```text
scripts/tdelight_segment_cmd.py
```

Local smoke environment:

```text
dev-ignore/venvs/tdelight-py312
dev-ignore/vendor/turkish-delight-nlp-api
```

Reason: TurkishDelightNLP is a Python-3.7-era DyNet project. The local runtime
uses a Python 3.12 virtualenv with a patched local DyNet 2.1.2 build, the
DyNet-pinned Eigen archive, and a small Gensim-4 import compatibility shim in
the ignored vendor checkout.

Smoke examples:

| Word | Pieces |
| --- | --- |
| `evlerden` | `ev + ler + den` |
| `geldik` | `gel + di + k` |
| `kitaplarım` | `kitap + lar + ım` |
| `çalışıyorum` | `çalış + ıyor + um` |
| `evlerimizden` | `ev + ler + im + iz + den` |

Corrected hash-100k screening result:

```text
dev-ignore/morph-smoke/benchmarks/tdelight_hash_100k.json
```

| Metric | Value |
| --- | ---: |
| Processed unique word forms | 100,000 |
| Processed weighted words | 12,095,210 |
| Runtime | 65.29 s |
| Unique forms / second | 1,532 |
| Pieces / word, weighted | 1.503 |
| Split rate, weighted | 0.362 |
| Fallback rate, weighted | 0.000 |
| Fallback rate, types | 0.000 |

The wrapper maps TurkishDelight's lowercased/Unicode-normalized outputs back to
exact original surface slices before the benchmark parser validates them. This
keeps the no-cross-boundary invariant while avoiding false fallbacks from
uppercase `İ` and curly apostrophe normalization.

The repo adapter can use either:

- `TDELIGHT_SEGMENT_CMD`, a stdin/stdout command with one word per line, or
- `TDELIGHT_URL`, a REST service endpoint.

## Bounded Protocol

The full first shard has 10.83M unique word forms, so not every segmenter should
be forced through a full pass before we know its throughput. The benchmark tool
now supports deterministic unique-type limits while preserving full corpus
inventory in the output JSON.

Recommended stages:

| Stage | Selection | Unique type cap | Purpose |
| --- | --- | ---: | --- |
| Smoke | `hash` | 1,000 | Verify backend wiring and JSON shape. |
| Screening | `hash` | 100,000 | Estimate coverage, split rate, fallback rate, and throughput. |
| Frequency check | `frequency` | 100,000 | Measure behavior on common high-impact forms. |
| Candidate run | `hash` or full | 1,000,000+ | Promote only fast, low-fallback tools. |
| Full shard | `lexical`, no cap | 10,829,544 | Use only when throughput and memory behavior are acceptable. |

Example bounded command:

```bash
NANOCHAT_BASE_DIR=/Users/nurcunal/Documents/nanochat-turk/dev-ignore/morph-smoke \
python3 -m scripts.morph_benchmark \
  --backend identity \
  --max-files 1 \
  --max-words 0 \
  --word-counts-cache dev-ignore/morph-smoke/benchmarks/word_counts_000_00000.pkl \
  --max-unique-words 100000 \
  --word-selection hash \
  --output dev-ignore/morph-smoke/benchmarks/identity_hash_100k.json
```

TRmorph bounded command:

```bash
TRMORPH_SEGMENT_FST=/private/tmp/TRmorph/segment.fst \
TRMORPH_FLOOKUP_FLAGS=-x \
TRMORPH_MAX_OUTPUT_LINES_PER_WORD=512 \
TRMORPH_MAX_ANALYSES_PER_WORD=64 \
NANOCHAT_BASE_DIR=/Users/nurcunal/Documents/nanochat-turk/dev-ignore/morph-smoke \
python3 -m scripts.morph_benchmark \
  --backend trmorph \
  --max-files 1 \
  --max-words 0 \
  --word-counts-cache dev-ignore/morph-smoke/benchmarks/word_counts_000_00000.pkl \
  --max-unique-words 100000 \
  --word-selection hash \
  --batch-size 2048 \
  --timeout 60 \
  --output dev-ignore/morph-smoke/benchmarks/trmorph_hash_100k.json
```

Zemberek bounded command:

```bash
ZEMBEREK_SEGMENT_CMD="/private/tmp/zemberek-smoke-py312/bin/python scripts/zemberek_segment_cmd.py" \
NANOCHAT_BASE_DIR=/Users/nurcunal/Documents/nanochat-turk/dev-ignore/morph-smoke \
python3 -m scripts.morph_benchmark \
  --backend zemberek \
  --max-files 1 \
  --max-words 0 \
  --word-counts-cache dev-ignore/morph-smoke/benchmarks/word_counts_000_00000.pkl \
  --max-unique-words 100000 \
  --word-selection hash \
  --batch-size 100000 \
  --timeout 900 \
  --output dev-ignore/morph-smoke/benchmarks/zemberek_hash_100k.json
```

TurkishDelightNLP bounded command:

```bash
TDELIGHT_SEGMENT_CMD="dev-ignore/venvs/tdelight-py312/bin/python scripts/tdelight_segment_cmd.py" \
NANOCHAT_BASE_DIR=/Users/nurcunal/Documents/nanochat-turk/dev-ignore/morph-smoke \
python3 -m scripts.morph_benchmark \
  --backend tdelight \
  --max-files 1 \
  --max-words 0 \
  --word-counts-cache dev-ignore/morph-smoke/benchmarks/word_counts_000_00000.pkl \
  --max-unique-words 100000 \
  --word-selection hash \
  --batch-size 4096 \
  --timeout 300 \
  --output dev-ignore/morph-smoke/benchmarks/tdelight_hash_100k.json
```
