# TurkishDelightNLP Local Setup Notes

TurkishDelightNLP is now wired into the morphology benchmark pipeline through:

```text
scripts/tdelight_segment_cmd.py
```

The wrapper reads one word per stdin line and writes one JSON list of surface
morpheme pieces per stdout line. Configure it with:

```bash
export TDELIGHT_SEGMENT_CMD="dev-ignore/venvs/tdelight-py312/bin/python scripts/tdelight_segment_cmd.py"
```

## Local Runtime

The local test runtime is intentionally outside git:

```text
dev-ignore/vendor/turkish-delight-nlp-api
dev-ignore/venvs/tdelight-py312
dev-ignore/vendor/dynet-eigen/eigen-b2e267dc99d4
dev-ignore/vendor/dynet-sdist/dyNET-2.1.2
```

Upstream repo used locally:

```text
https://github.com/halecakir/turkish-delight-nlp-api
```

Model files used by the wrapper:

```text
data/Turkish-jointAll-MTAG_COMP=w_sum-MORPH_COMP=w_sum-POS_COMP=w_sum-COMP_ALPHA=0.1-trialmodel
data/Turkish-jointAll-MTAG_COMP=w_sum-MORPH_COMP=w_sum-POS_COMP=w_sum-COMP_ALPHA=0.1-trialmodel.params
```

## Compatibility Notes

The upstream project targets an older Python/DyNet stack. The local Python 3.12
runtime required these ignored vendor fixes:

- Build DyNet 2.1.2 from local source rather than current PyPI wheels.
- Use DyNet's pinned Eigen archive instead of Homebrew Eigen 5.
- Add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` for modern CMake.
- Compile DyNet with C++14 locally.
- Install `gensim` and make `gensim.models.wrappers.FastText` optional in the
  ignored vendor checkout because Gensim 4 removed that module.

The tracked wrapper also repairs harmless surface-normalization differences by
mapping TurkishDelight's output pieces back onto exact slices of the original
word. This is needed for uppercase `İ`, combining-dot output, and curly
apostrophe normalization.

## Smoke Test

```bash
printf 'evlerden\ngeldik\nkitaplarım\nçalışıyorum\nevlerimizden\n' \
  | dev-ignore/venvs/tdelight-py312/bin/python scripts/tdelight_segment_cmd.py
```

Expected stdout pieces:

```text
["ev", "ler", "den"]
["gel", "di", "k"]
["kitap", "lar", "ım"]
["çalış", "ıyor", "um"]
["ev", "ler", "im", "iz", "den"]
```

DyNet and model-load messages may appear on stderr.

## Benchmark Command

```bash
PYTHONPATH=. \
TDELIGHT_SEGMENT_CMD="dev-ignore/venvs/tdelight-py312/bin/python scripts/tdelight_segment_cmd.py" \
python3 scripts/morph_benchmark.py \
  --backend tdelight \
  --data-dir dev-ignore/morph-smoke/base_data_fineweb2_tur_latn \
  --max-files 1 \
  --max-words 0 \
  --max-docs 0 \
  --word-counts-cache dev-ignore/morph-smoke/benchmarks/word_counts_000_00000.pkl \
  --word-selection hash \
  --max-unique-words 100000 \
  --batch-size 4096 \
  --timeout 300 \
  --output dev-ignore/morph-smoke/benchmarks/tdelight_hash_100k.json
```

Current hash-100k result:

| Metric | Value |
| --- | ---: |
| Runtime | 65.29 s |
| Unique forms / second | 1,532 |
| Pieces / word, weighted | 1.503 |
| Split rate, weighted | 0.362 |
| Fallback rate, weighted | 0.000 |
