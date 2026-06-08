# Codex-Local Segmenter Judge Results

This is the first no-API judge pass over the Turkish morphology segmenter
outputs. It uses Codex locally as the judge/rubric author and does not call an
external LLM service.

## Artifacts

Tracked outputs:

```text
docs/tokenizer_tests/judge_results/codex_local_hash_100k_judge_500.judgments.jsonl
docs/tokenizer_tests/judge_results/codex_local_hash_100k_judge_500.scores.json
```

Local blind input pack:

```text
dev-ignore/morph-smoke/judge_packs/hash_100k_judge_500/hash_100k_judge_500.jsonl
```

Scoring answer key, kept local:

```text
dev-ignore/morph-smoke/judge_packs/hash_100k_judge_500/hash_100k_judge_500.answer_key.json
```

## Methodology

The judge pack contains `500` disagreement-focused words selected from the same
deterministic `hash` sample of `100,000` unique word forms from the first
FineWeb-2 Turkish shard. Candidate labels were blind (`A`, `B`, `C`, `D`) and
were decoded to backend names only after all item-level judgments were written.

For each word, the judge selected:

- `best_label`: the single best segmentation among the blind candidates.
- `acceptable_labels`: all candidates that were linguistically acceptable or
  tied closely with the best candidate.
- `confidence`: `high`, `medium`, or `low`.

Rubric:

1. Preserve the exact original surface word when pieces are concatenated.
2. Prefer plausible Turkish stem plus suffix boundaries.
3. Prefer segmentations that expose productive morphemes such as plural, case,
   possessive, derivational, passive, causative, ability, tense/aspect, and
   copular suffixes.
4. Penalize splits that cut through stems, such as `bek+ler+ken` for
   `beklerken`.
5. Penalize over-segmentation of non-morphemic buffer material, such as
   `dese+n+ler+i` for `desenleri`.
6. Allow identity/no-split mainly for noisy, foreign, proper-name, all-caps, or
   ambiguous cases.

The reproducible local judge implementation is:

```text
scripts/morph_codex_local_judge.py
```

It reads only the blind JSONL. Backend names are introduced afterward by:

```bash
python3 -m scripts.morph_judge_score \
  --judgments docs/tokenizer_tests/judge_results/codex_local_hash_100k_judge_500.judgments.jsonl \
  --answer-key dev-ignore/morph-smoke/judge_packs/hash_100k_judge_500/hash_100k_judge_500.answer_key.json \
  --output docs/tokenizer_tests/judge_results/codex_local_hash_100k_judge_500.scores.json
```

## Scores

| Backend | Best count | Best rate | Acceptable count | Acceptable rate |
| --- | ---: | ---: | ---: | ---: |
| TRmorph | 241 | 48.2% | 255 | 51.0% |
| TurkishDelightNLP | 173 | 34.6% | 219 | 43.8% |
| Zemberek | 86 | 17.2% | 200 | 40.0% |
| identity | 0 | 0.0% | 4 | 0.8% |

Confidence distribution:

| Confidence | Count | Rate |
| --- | ---: | ---: |
| high | 148 | 29.6% |
| medium | 287 | 57.4% |
| low | 65 | 13.0% |

## Interpretation

TRmorph wins the most blind best-label decisions. Its strongest cases are words
where exposing deeper Turkish morphology matters, such as passive,
derivational, ability, participial, or locative `+ki` boundaries.

TurkishDelightNLP is a strong second and has very broad surface coverage. It
often wins clean inflectional cases, but the judge penalized cases where it
kept derivational/passive material fused or split person/possessive chunks too
finely.

Zemberek is frequently acceptable but less often best. In this parser/wrapper
setup it tends to be more conservative, often preserving fused suffix groups or
falling back to identity when analyzer output cannot be converted to exact
surface pieces.

Identity is not competitive on this disagreement-focused Turkish morphology
sample, which is expected. It remains useful only as a control and for noisy
tokens where all segmenters hallucinate boundaries.

## Recommendation

Carry at least TRmorph and TurkishDelightNLP forward into tokenizer training
ablations. TRmorph appears linguistically sharper when it produces a usable
segmentation; TurkishDelightNLP has better coverage and zero fallback in the
current wrapper. A hybrid segmenter is worth testing: prefer TRmorph when it
returns a non-fallback surface segmentation, otherwise fall back to
TurkishDelightNLP.

Zemberek should remain as a conservative control and as a possible ingredient
for a later hybrid, but the current blind pass does not support making it the
primary segmenter.

## Limitations

This is a preliminary Codex-local judge pass, not a native-speaker gold
annotation. The current pack has no context snippets, so genuinely ambiguous
forms can only be judged by likely morphology. The sample is
disagreement-focused, not corpus-representative. Also, when two tools emit the
same segmentation, `acceptable_labels` is more informative than the single
`best_label`.
