# CETVEL Core-12 Model Comparison

Compact comparison of the completed base-model CETVEL core slice for the raw
BPE baseline and the available MorphBPE variant runs.

## Scope

- Date archived: 2026-06-12
- Model step: `17100` for all rows
- Model depth/vocab tier: d20, 32k tokenizer family
- Compared slice: CETVEL core tasks 01-12
- Macro summary: mean over the 11 classification/loglikelihood tasks, excluding
  `xquad_tr` because it reports F1 on a different scale
- Benchmark speed: progress-log core-12 `total_elapsed` divided by `39,441`
  expanded effective examples
- Final validation BPB source: final checkpoint `meta_017100.json`
- Lowest validation BPB source: `loop_state.min_val_bpb` in final checkpoint
  metadata
- Final train loss source: last printed training step, `step 17099/17100`

The raw BPE run has a larger checked-in artifact for tasks 01-13 under
[`artifacts/cetvel_base_subset_2026-06-09_job493293`](../cetvel_base_subset_2026-06-09_job493293/).
The comparison here uses only the common task slice so the model rows are
comparable.

## Run Summary

| Run | Tokenizer | CETVEL job | Slice | Core-12 elapsed | CETVEL ex/s up | Speed vs raw | Final val BPB | Lowest val BPB | Final train loss | Core-11 macro | Delta vs raw BPE | XQuAD F1 | Delta vs raw BPE |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw BPE d20 | `bpe_32768` | `493293` | tasks 01-13 archived; common tasks 01-12 used here | 50m20s | 13.06 | 1.000x | 0.6232 | 0.6232 | 2.4899 | 0.4514 | +0.0000 | 3.0985 | +0.0000 |
| MorphBPE + TRmorph d20 | `morphbpe_trmorph_32768` | `494056` | core tasks 01-12 | 52m38s | 12.49 | 0.956x | 0.6266 | 0.6266 | 2.0106 | 0.4541 | +0.0027 | 3.4786 | +0.3801 |
| MorphBPE + Zemberek d20 | `morphbpe_zemberek_32768` | `494057` | core tasks 01-12 | 50m30s | 13.02 | 0.997x | 0.6250 | 0.6250 | 2.3227 | 0.4618 | +0.0104 | 3.2633 | +0.1648 |

## Task Metrics

| Task | Metric | Raw BPE | TRmorph MorphBPE | Delta | Zemberek MorphBPE | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `exams_tr` | `acc_norm` | 0.3104 | 0.2875 | -0.0229 | 0.2952 | -0.0153 |
| `belebele_tr` | `acc_norm` | 0.2522 | 0.2356 | -0.0167 | 0.2433 | -0.0089 |
| `turkish_plu` | `acc_norm` | 0.5027 | 0.5146 | +0.0118 | 0.5011 | -0.0016 |
| `cetvel_xcopa_tr` | `acc` | 0.6180 | 0.6080 | -0.0100 | 0.6220 | +0.0040 |
| `cetvel_xnli_tr` | `acc_norm` | 0.3335 | 0.3301 | -0.0034 | 0.3325 | -0.0010 |
| `mnli_tr` | `acc_norm` | 0.3210 | 0.3215 | +0.0005 | 0.3216 | +0.0006 |
| `snli_tr` | `acc_norm` | 0.3234 | 0.3235 | +0.0001 | 0.3195 | -0.0039 |
| `news_cat` | `acc_norm` | 0.6760 | 0.7320 | +0.0560 | 0.7240 | +0.0480 |
| `offenseval_tr` | `acc_norm` | 0.7971 | 0.7764 | -0.0207 | 0.7937 | -0.0034 |
| `trclaim19` | `acc_norm` | 0.4938 | 0.5283 | +0.0345 | 0.6010 | +0.1072 |
| `xfact_tr` | `acc_norm` | 0.3373 | 0.3373 | +0.0000 | 0.3254 | -0.0118 |
| `xquad_tr` | `f1` | 3.0985 | 3.4786 | +0.3801 | 3.2633 | +0.1648 |

## Source Results

```text
raw_bpe:
  /ari/users/nunal/nanochat-turk-d20-bpe32k/cetvel_out/full_kvcache_20260609_185114/cetvel_full_partial_results.json

morphbpe_trmorph:
  /ari/users/nunal/nanochat-turk-morphbpe-trmorph-32768/cetvel_out/core_tasks_01_12/cetvel_core_results.json

morphbpe_zemberek:
  /ari/users/nunal/nanochat-turk-morphbpe-zemberek-32768/cetvel_out/core_tasks_01_12/cetvel_core_results.json
```

## Source Timing Logs

```text
raw_bpe:
  /ari/users/nunal/nanochat-turk/nanochat-cetvel-full-493293.out

morphbpe_trmorph:
  /ari/users/nunal/nanochat-turk/cetvel-core12-trmorph-d20-494056.out

morphbpe_zemberek:
  /ari/users/nunal/nanochat-turk/cetvel-core12-zemberek-d20-494057.out
```

## Interpretation

The core-11 macro moves only modestly from raw BPE to TRmorph MorphBPE
(`+0.0027`) and more clearly for Zemberek MorphBPE (`+0.0104`). The gains are
task-specific rather than uniform: MorphBPE improves `news_cat` and
`trclaim19`, while raw BPE remains stronger on tasks such as `exams_tr`,
`belebele_tr`, and `offenseval_tr`. TRmorph has the best XQuAD F1 in this slice.
Raw BPE has the best final and lowest validation BPB. The lowest value equals
the final value for all rows because validation kept improving through step
`17100`. Final train loss is included as run telemetry, but validation BPB is
the comparable loss metric across tokenizers. The benchmark-speed column is
nearly tied for raw BPE and Zemberek MorphBPE, with TRmorph MorphBPE about 4%
slower end-to-end in this matched harness.

Use this table as early model-facing evidence, not as a final tokenizer verdict.
The runs are pre-SFT base models and the generation-heavy CETVEL tasks are not
post-SFT assistant-quality measurements.
