# CETVEL Model Comparison

This page tracks model-facing CETVEL evidence for the tokenizer ablation. It is
separate from tokenizer-only diagnostics: these rows compare trained d20 base
models at the same checkpoint step.

## Current Comparable Slice

The completed comparable slice is CETVEL core tasks 01-12 for:

- raw BPE baseline, `tr_d20_bpe_32768_chinchilla20`;
- TRmorph MorphBPE, `tr_d20_morphbpe_trmorph_32768_chinchilla20`;
- Zemberek MorphBPE, `tr_d20_morphbpe_zemberek_32768_chinchilla20`.

All rows use model step `17100`. Validation BPB comes from the final checkpoint
metadata, while final train loss comes from the last printed training step
(`step 17099/17100`). The raw BPE artifact includes tasks 01-13, but the
comparison below uses only tasks 01-12 because that is the common completed
slice for the MorphBPE runs.

The macro score averages the 11 classification/loglikelihood tasks. `xquad_tr`
is reported separately because it is F1 rather than accuracy.

| Run | Tokenizer | Segmenter | CETVEL job | Val BPB | Final train loss | Core-11 macro | Delta vs raw BPE | XQuAD F1 | Delta vs raw BPE | Source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Raw BPE d20 | `bpe_32768` | none | `493293` | 0.6232 | 2.4899 | 0.4514 | +0.0000 | 3.0985 | +0.0000 | [raw artifact](../artifacts/cetvel_base_subset_2026-06-09_job493293/) |
| MorphBPE + TRmorph d20 | `morphbpe_trmorph_32768` | TRmorph | `494056` | 0.6266 | 2.0106 | 0.4541 | +0.0027 | 3.4786 | +0.3801 | [comparison artifact](../artifacts/cetvel_core12_model_comparison_2026-06-12/) |
| MorphBPE + Zemberek d20 | `morphbpe_zemberek_32768` | Zemberek | `494057` | 0.6250 | 2.3227 | 0.4618 | +0.0104 | 3.2633 | +0.1648 | [comparison artifact](../artifacts/cetvel_core12_model_comparison_2026-06-12/) |

## Per-Task Core Comparison

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

## Reading The Result

This is the first checked-in model-facing comparison between the raw baseline
and MorphBPE variants. It does not fully settle the tokenizer question.

- TRmorph MorphBPE slightly improves the core-11 macro and has the best XQuAD
  F1 in the common slice.
- Zemberek MorphBPE has the strongest core-11 macro, driven especially by
  `trclaim19` and `news_cat`.
- Raw BPE remains better on several tasks, including `exams_tr`, `belebele_tr`,
  and `offenseval_tr`.
- The differences are task-specific, so the report should avoid claiming a
  uniform MorphBPE win from this table alone.
- Raw BPE has the best final validation BPB in this first d20 slice. Final train
  loss is useful telemetry but should not be used as the cross-tokenizer loss
  metric because tokenization changes the prediction units.

The next report-critical step is to keep the model comparison table synchronized
as TurkishDelightNLP and any full-CETVEL or post-SFT runs finish.
