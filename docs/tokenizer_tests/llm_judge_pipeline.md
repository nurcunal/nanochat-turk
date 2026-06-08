# LLM Judge Pipeline For Segmenter Quality

This pipeline creates an equal-standard sample across segmenters, exports a
small blind judge pack, and scores the LLM/human judgments afterward.

Large generated packs live locally under `dev-ignore/` and are not committed.

## Equal Standard

Default screening standard:

- Source: first FineWeb-2 Turkish shard, `000_00000.parquet`.
- Inventory: full shard word-count cache
  `dev-ignore/morph-smoke/benchmarks/word_counts_000_00000.pkl`.
- Sample: deterministic `hash` sample of `100,000` unique word forms.
- Judge subset: `500` disagreement-focused items from that same 100k sample.
- Candidates: identity/no-split control, TRmorph, Zemberek, and TurkishDelight
  when configured.
- Context: corpus snippets are included when found within the configured scan
  budget.

The 100k metric sample is for quantitative comparison. The 500-item judge pack
is for quality comparison. This is large enough to see systematic differences
without asking an LLM to judge 100k rows.

## Generate Pack

```bash
TRMORPH_SEGMENT_FST=/private/tmp/TRmorph/segment.fst \
TRMORPH_FLOOKUP_FLAGS=-x \
TRMORPH_MAX_OUTPUT_LINES_PER_WORD=512 \
TRMORPH_MAX_ANALYSES_PER_WORD=64 \
ZEMBEREK_SEGMENT_CMD="/private/tmp/zemberek-smoke-py312/bin/python scripts/zemberek_segment_cmd.py" \
TDELIGHT_SEGMENT_CMD="dev-ignore/venvs/tdelight-py312/bin/python scripts/tdelight_segment_cmd.py" \
NANOCHAT_BASE_DIR=/Users/nurcunal/Documents/nanochat-turk/dev-ignore/morph-smoke \
python3 -m scripts.morph_judge_pack \
  --backend trmorph \
  --backend zemberek \
  --backend tdelight \
  --include-identity \
  --max-files 1 \
  --max-words 0 \
  --word-counts-cache dev-ignore/morph-smoke/benchmarks/word_counts_000_00000.pkl \
  --max-unique-words 100000 \
  --word-selection hash \
  --judge-size 500 \
  --judge-strategy disagreement \
  --batch-size 2048 \
  --backend-batch-size trmorph=2048 \
  --backend-batch-size zemberek=100000 \
  --backend-batch-size tdelight=4096 \
  --timeout 900 \
  --output-dir dev-ignore/morph-smoke/judge_packs/hash_100k_judge_500 \
  --prefix hash_100k_judge_500
```

Add `--include-context --context-max-docs 250000` when the judge pack should
include corpus snippets. The current lightweight pack omits context so it can
be uploaded easily.

Generated local files:

```text
dev-ignore/morph-smoke/judge_packs/hash_100k_judge_500/
  hash_100k_judge_500.jsonl
  hash_100k_judge_500.answer_key.json
  hash_100k_judge_500.metrics.json
  hash_100k_judge_500.prompt.md
```

Upload only:

- `hash_100k_judge_500.prompt.md`
- `hash_100k_judge_500.jsonl`

Keep local until after judging:

- `hash_100k_judge_500.answer_key.json`
- `hash_100k_judge_500.metrics.json`

The judge JSONL is blind: candidates are labeled `A`, `B`, `C`, etc. The answer
key maps labels back to segmenters after judging.

## Current Equal-100k Results

The current run used identity, TRmorph, Zemberek, and TurkishDelightNLP.

| Backend | Status | Unique/s | Weighted split rate | Weighted fallback | Type fallback |
| --- | --- | ---: | ---: | ---: | ---: |
| identity | ok | 510,725 | 0.000 | 0.000 | 0.000 |
| TRmorph | ok | 9,351 | 0.338 | 0.145 | 0.817 |
| Zemberek | ok | 3,296 | 0.316 | 0.425 | 0.233 |
| TurkishDelightNLP | ok | 1,509 | 0.362 | 0.000 | 0.000 |

Judge-pack details:

- Judge rows: `500`
- Rows with context found: `0` in the current lightweight pack.
- Local judge JSONL size: about `260 KiB`

Interpretation before LLM judging:

- TRmorph is faster and has much lower frequency-weighted fallback.
- Zemberek has much lower type fallback but higher frequency-weighted fallback
  in this parser/wrapper setup.
- TurkishDelightNLP has zero fallback after surface-realignment repair, but is
  slower and tends to split more unique types than TRmorph or Zemberek.
- The high TRmorph type fallback is mostly rare noisy web types; the lower
  weighted fallback suggests it covers frequent Turkish forms better.
- LLM/human judging should decide whether TRmorph's, Zemberek's, or
  TurkishDelightNLP's splits are linguistically best on real Turkish forms.

## Score Judgments

After the LLM returns JSONL in the requested format, score it with:

```bash
python3 -m scripts.morph_judge_score \
  --judgments dev-ignore/morph-smoke/judge_packs/hash_100k_judge_500/llm_judgments.jsonl \
  --answer-key dev-ignore/morph-smoke/judge_packs/hash_100k_judge_500/hash_100k_judge_500.answer_key.json \
  --output dev-ignore/morph-smoke/judge_packs/hash_100k_judge_500/llm_judgment_scores.json
```

The scorer reports best-label counts/rates by backend, acceptable-label counts,
confidence counts, and invalid rows.

## TurkishDelightNLP

The command wrapper is:

```bash
export TDELIGHT_SEGMENT_CMD="dev-ignore/venvs/tdelight-py312/bin/python scripts/tdelight_segment_cmd.py"
```

The repo adapter can also use a REST service endpoint:

```bash
export TDELIGHT_URL="http://localhost:PORT"
```
