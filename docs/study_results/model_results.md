# Pretraining and model benchmarks

**The original model benchmark table needs a protocol qualification.** Raw BPE has the lowest archived validation BPB in all three tiers, subject to unmatched raw-text evaluation spans. The saved core-11 table puts Zemberek ahead by 1.04 percentage points at 32k, but recovered evaluator code documents generation and continuation-boundary errors. Treat those benchmark rankings as legacy measurements pending a corrected eleven-model rerun; do not present them as established tokenizer effects.

Report assembled and audited against UHeM on 2026-11-09. All 132 archived task cells match recovered raw results at four decimals. Ten original step-17,100 metadata files remain; the raw-BPE 32k metadata at its original path is absent, so that BPB retains archived provenance. Eleven of twelve model cells have results; 32k TurkishDelight remains tokenizer-only. No new model training or benchmark evaluation was performed for this report.

**Correction evidence:** the same d20/raw-BPE step-17,100 checkpoint scores XQuAD F1 **11.3369** with corrected generation handling, versus the archived **3.0985**. Corrected results for all MorphBPE models were not recovered. See [later model results and evaluator audit](later_model_results.md) for the separate d20 continuation, d32 training, and protocol-specific benchmarks.

Download [model summary CSV](model_results.csv), [training statistics CSV](model_training_statistics.csv), [all 132 task scores CSV](model_task_metrics.csv), or [full JSON with source paths and hashes](model_results.json). Original evidence: [June 22 benchmark artifact](../../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/metrics_summary.json).

## Experiment controls

Compare tokenizer families **within a vocabulary tier**. The documented design holds the raw FineWeb-2 Turkish source/manifest/order, model shape, optimizer recipe, token-position budget and evaluation harness fixed within that tier. Across tiers, depth changes to keep parameter counts near 0.9B.

| Vocab size | Transformer depth | Approx. parameters | Steps | Token positions/step | Total token positions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 20 | 896.5M | 17,100 | 1,048,576 | 17,930,649,600 |
| 65,536 | 16 | 872.4M | 17,100 | 1,048,576 | 17,930,649,600 |
| 131,072 | 12 | 890.2M | 17,100 | 1,048,576 | 17,930,649,600 |

Recovered original reports/configurations document four A100 GPUs and 2,048-token context. Ten original checkpoint metadata files survive; the raw-BPE32k run is supported by its historical training log and archived metadata values. The ten original report headers record Python 3.12.4 and PyTorch 2.9.1+cu128, but their Git revision is unknown. Source initialization uses seed 42; complete per-run code/environment fingerprints and repeated-seed experiments remain absent. See the [design](../tokenizer_ablation_plan.md) and [launch record](../tokenizer_tests/vocab64_128_uhem_launch.md).

Fixed token positions do **not** mean equal raw bytes, documents, or characters of context. The compact results lack per-run exposure counts. Validation also uses a fixed token budget, so BPB normalization alone does not guarantee the exact same raw-text evaluation spans. Cross-tier results combine vocabulary allocation and model-depth effects.

## Pretraining results

BPB is bits per UTF-8 byte after excluding special tokens; lower is better. Final BPB equals the recorded minimum in all eleven completed runs. Training loss is the last printed token-level loss; it is preserved for completeness but cannot be compared directly across different tokenizations. All completed models end at step 17,100.

| Tier | Tokenizer | Final/minimum val BPB ↓ | Δ BPB vs raw | Final token loss | Status |
| --- | --- | ---: | ---: | ---: | --- |
| 32k/d20 | Raw BPE | 0.6232 | +0.0000 | 2.4899 | Complete |
| 32k/d20 | MorphBPE · TRmorph | 0.6266 | +0.0034 | 2.0106 | Complete |
| 32k/d20 | MorphBPE · Zemberek | 0.6250 | +0.0018 | 2.3227 | Complete |
| 32k/d20 | MorphBPE · TurkishDelight | — | — | — | No full model in archived inventory |
| 64k/d16 | Raw BPE | 0.6409 | +0.0000 | 2.5812 | Complete |
| 64k/d16 | MorphBPE · TRmorph | 0.6521 | +0.0112 | 2.2754 | Complete |
| 64k/d16 | MorphBPE · Zemberek | 0.6514 | +0.0105 | 2.3013 | Complete |
| 64k/d16 | MorphBPE · TurkishDelight | 0.6510 | +0.0101 | 2.4869 | Complete |
| 128k/d12 | Raw BPE | 0.6749 | +0.0000 | 3.0976 | Complete |
| 128k/d12 | MorphBPE · TRmorph | 0.6917 | +0.0168 | 2.5947 | Complete |
| 128k/d12 | MorphBPE · Zemberek | 0.6940 | +0.0191 | 2.6000 | Complete |
| 128k/d12 | MorphBPE · TurkishDelight | 0.6820 | +0.0071 | 2.6477 | Complete |

## Recovered training cost and hardware metrics

Training-loop time is separate from Slurm job elapsed time and CETVEL benchmark time. All eleven original jobs were recovered as `COMPLETED`, exit `0:0`; original training used four A100 GPUs. FLOPs and MFU are implementation estimates. Cross-tier speed also changes model depth and cannot isolate tokenizer efficiency.

| Tier | Tokenizer | Parameters | Train minutes | Slurm elapsed | Estimated FLOPs | Peak MiB | MFU % |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 32k/d20 | Raw BPE | 896,533,746 | 1267.66 | 21:31:38 | 5.809722e+19 | 18499.62 | — |
| 32k/d20 | MorphBPE · TRmorph | 896,533,746 | 1265.77 | 21:30:53 | 5.809722e+19 | 18499.74 | 61.29 |
| 32k/d20 | MorphBPE · Zemberek | 896,533,746 | 1270.43 | 21:35:34 | 5.809722e+19 | 18499.62 | 61.11 |
| 64k/d16 | Raw BPE | 872,416,058 | 907.39 | 15:30:31 | 3.609925e+19 | 14645.31 | 53.14 |
| 64k/d16 | MorphBPE · TRmorph | 872,416,058 | 905.97 | 15:28:04 | 3.609925e+19 | 14645.31 | 53.20 |
| 64k/d16 | MorphBPE · Zemberek | 872,416,058 | 905.82 | 15:27:54 | 3.609925e+19 | 14645.31 | 53.22 |
| 64k/d16 | MorphBPE · TurkishDelight | 872,416,058 | 903.50 | 15:25:56 | 3.609925e+19 | 14645.31 | 53.41 |
| 128k/d12 | Raw BPE | 890,241,506 | 663.68 | 11:26:01 | 2.402855e+19 | 13066.91 | 48.35 |
| 128k/d12 | MorphBPE · TRmorph | 890,241,506 | 658.31 | 11:19:34 | 2.402855e+19 | 13065.79 | 48.81 |
| 128k/d12 | MorphBPE · Zemberek | 890,241,506 | 658.23 | 11:19:37 | 2.402855e+19 | 13068.04 | 48.85 |
| 128k/d12 | MorphBPE · TurkishDelight | 890,241,506 | 659.88 | 11:21:12 | 2.402855e+19 | 13066.91 | 48.69 |

The raw-BPE32k report directory was overwritten by its later continuation. Its original 1,267.66-minute training time and memory/FLOP values come from job 492421's surviving log, not that overwritten report. The recovered excerpt does not include its final aggregate MFU, so that cell remains unavailable. Source text and job accounting are retained in the [November audit artifact](../../artifacts/uhem_model_audit_2026-11-09/).

## Archived common CETVEL benchmark — legacy evaluator

These original values are retained, not silently replaced with results from a different evaluator. The legacy adapter ignored generation kwargs and used a separately tokenized context length without validating the continuation boundary. Both can affect comparisons across tokenizers; the complete matrix needs a corrected rerun. Core-11 is the unweighted mean of eleven classification/log-likelihood task scores on a 0–1 scale. XQuAD F1 uses a 0–100 scale and is excluded from that mean. Core-11 deltas below are **percentage points**; XQuAD deltas are F1 points. Timing divides 39,441 expanded effective examples by the logged core-12 elapsed time.

| Tier | Tokenizer | Core-11 ↑ | Δ pp | XQuAD F1 ↑ | Δ F1 | Elapsed | ex/s ↑ | Speed/raw | Job |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 32k/d20 | Raw BPE | 0.4514 | +0.00 | 3.0985 | +0.0000 | 50m20s | 13.06 | 1.000× | 493293 |
| 32k/d20 | MorphBPE · TRmorph | 0.4541 | +0.27 | 3.4786 | +0.3801 | 52m38s | 12.49 | 0.956× | 494056 |
| 32k/d20 | MorphBPE · Zemberek | 0.4618 | +1.04 | 3.2633 | +0.1648 | 50m30s | 13.02 | 0.997× | 494057 |
| 32k/d20 | MorphBPE · TurkishDelight | — | — | — | — | — | — | — | — |
| 64k/d16 | Raw BPE | 0.4590 | +0.00 | 2.8576 | +0.0000 | 43m27s | 15.13 | 1.000× | 496898 |
| 64k/d16 | MorphBPE · TRmorph | 0.4532 | -0.58 | 3.2778 | +0.4202 | 43m09s | 15.23 | 1.007× | 496899 |
| 64k/d16 | MorphBPE · Zemberek | 0.4568 | -0.22 | 2.8956 | +0.0380 | 42m45s | 15.38 | 1.016× | 496900 |
| 64k/d16 | MorphBPE · TurkishDelight | 0.4567 | -0.23 | 3.3280 | +0.4704 | 40m52s | 16.09 | 1.063× | 496901 |
| 128k/d12 | Raw BPE | 0.4651 | +0.00 | 2.2674 | +0.0000 | 35m13s | 18.67 | 1.000× | 496902 |
| 128k/d12 | MorphBPE · TRmorph | 0.4503 | -1.48 | 2.3517 | +0.0843 | 35m09s | 18.70 | 1.002× | 496903 |
| 128k/d12 | MorphBPE · Zemberek | 0.4618 | -0.33 | 2.9685 | +0.7011 | 35m19s | 18.61 | 0.997× | 496904 |
| 128k/d12 | MorphBPE · TurkishDelight | 0.4481 | -1.70 | 2.2498 | -0.0176 | 33m25s | 19.67 | 1.054× | 496905 |

The low legacy XQuAD scores are affected by the generation bug and should not be used to judge absolute QA ability or tokenizer gains. Recovered raw task standard errors do not provide paired confidence intervals for model differences. The 128k TurkishDelight result file was complete at import time while Slurm was still finishing uploads; that archived job-state qualification is retained in JSON.

## Per-task legacy results: 32k

Legacy adapter values. Classification metrics use 0–1; the final XQuAD row uses 0–100. `acc_norm` is the benchmark's length-normalized choice score; it is not F1. Interpretation requires the evaluator caveat above.

| Task | Metric | Raw BPE | MorphBPE · TRmorph | MorphBPE · Zemberek |
| --- | --- | ---: | ---: | ---: |
| `exams_tr` | `acc_norm` | 0.3104 | 0.2875 | 0.2952 |
| `belebele_tr` | `acc_norm` | 0.2522 | 0.2356 | 0.2433 |
| `turkish_plu` | `acc_norm` | 0.5027 | 0.5146 | 0.5011 |
| `cetvel_xcopa_tr` | `acc` | 0.6180 | 0.6080 | 0.6220 |
| `cetvel_xnli_tr` | `acc_norm` | 0.3335 | 0.3301 | 0.3325 |
| `mnli_tr` | `acc_norm` | 0.3210 | 0.3215 | 0.3216 |
| `snli_tr` | `acc_norm` | 0.3234 | 0.3235 | 0.3195 |
| `news_cat` | `acc_norm` | 0.6760 | 0.7320 | 0.7240 |
| `offenseval_tr` | `acc_norm` | 0.7971 | 0.7764 | 0.7937 |
| `trclaim19` | `acc_norm` | 0.4938 | 0.5283 | 0.6010 |
| `xfact_tr` | `acc_norm` | 0.3373 | 0.3373 | 0.3254 |
| `xquad_tr` | `f1` | 3.0985 | 3.4786 | 3.2633 |

## Per-task legacy results: 64k

Legacy adapter values. Classification metrics use 0–1; the final XQuAD row uses 0–100. `acc_norm` is the benchmark's length-normalized choice score; it is not F1. Interpretation requires the evaluator caveat above.

| Task | Metric | Raw BPE | MorphBPE · TRmorph | MorphBPE · Zemberek | MorphBPE · TurkishDelight |
| --- | --- | ---: | ---: | ---: | ---: |
| `exams_tr` | `acc_norm` | 0.3028 | 0.3079 | 0.2926 | 0.3028 |
| `belebele_tr` | `acc_norm` | 0.2433 | 0.2233 | 0.2300 | 0.2356 |
| `turkish_plu` | `acc_norm` | 0.4906 | 0.4928 | 0.4909 | 0.5046 |
| `cetvel_xcopa_tr` | `acc` | 0.6100 | 0.6160 | 0.6140 | 0.6100 |
| `cetvel_xnli_tr` | `acc_norm` | 0.3333 | 0.3333 | 0.3299 | 0.3335 |
| `mnli_tr` | `acc_norm` | 0.3213 | 0.3213 | 0.3064 | 0.3211 |
| `snli_tr` | `acc_norm` | 0.3237 | 0.3237 | 0.3209 | 0.3240 |
| `news_cat` | `acc_norm` | 0.6920 | 0.6400 | 0.7040 | 0.7000 |
| `offenseval_tr` | `acc_norm` | 0.7971 | 0.7968 | 0.7971 | 0.7971 |
| `trclaim19` | `acc_norm` | 0.5976 | 0.5932 | 0.6017 | 0.5583 |
| `xfact_tr` | `acc_norm` | 0.3373 | 0.3373 | 0.3373 | 0.3373 |
| `xquad_tr` | `f1` | 2.8576 | 3.2778 | 2.8956 | 3.3280 |

## Per-task legacy results: 128k

Legacy adapter values. Classification metrics use 0–1; the final XQuAD row uses 0–100. `acc_norm` is the benchmark's length-normalized choice score; it is not F1. Interpretation requires the evaluator caveat above.

| Task | Metric | Raw BPE | MorphBPE · TRmorph | MorphBPE · Zemberek | MorphBPE · TurkishDelight |
| --- | --- | ---: | ---: | ---: | ---: |
| `exams_tr` | `acc_norm` | 0.3104 | 0.2926 | 0.3079 | 0.2952 |
| `belebele_tr` | `acc_norm` | 0.2389 | 0.2400 | 0.2633 | 0.2233 |
| `turkish_plu` | `acc_norm` | 0.4653 | 0.4611 | 0.4893 | 0.4678 |
| `cetvel_xcopa_tr` | `acc` | 0.6000 | 0.5960 | 0.6180 | 0.6020 |
| `cetvel_xnli_tr` | `acc_norm` | 0.3325 | 0.3323 | 0.3329 | 0.3339 |
| `mnli_tr` | `acc_norm` | 0.3214 | 0.3209 | 0.3212 | 0.3210 |
| `snli_tr` | `acc_norm` | 0.3241 | 0.3234 | 0.3237 | 0.3243 |
| `news_cat` | `acc_norm` | 0.7280 | 0.6520 | 0.7120 | 0.6280 |
| `offenseval_tr` | `acc_norm` | 0.7971 | 0.7971 | 0.7971 | 0.7971 |
| `trclaim19` | `acc_norm` | 0.6021 | 0.6010 | 0.6003 | 0.5996 |
| `xfact_tr` | `acc_norm` | 0.3964 | 0.3373 | 0.3136 | 0.3373 |
| `xquad_tr` | `f1` | 2.2674 | 2.3517 | 2.9685 | 2.2498 |

## Historical evidence retained

The [June 12 comparison](../../artifacts/cetvel_core12_model_comparison_2026-06-12/) contains the same three 32k models. Shared summaries match the June 22 artifact at four decimal places, and all 36 task cells match exactly; these are not additional independent runs. Its higher-precision 32k BPB and train-loss values are preserved separately in JSON (raw-BPE BPB 0.6232126545832695, TRmorph 0.6265726039416546, Zemberek 0.6249791391940578).

The earlier [raw-BPE-only archive](../../artifacts/cetvel_base_subset_2026-06-09_job493293/metrics_summary.json) preserves additional metrics and subtasks. The following are baseline-only and must not be used as a tokenizer-family comparison:

| Baseline-only task | Metric | Value |
| --- | --- | ---: |
| `check_worthiness` | `acc` | 0.376143 |
| `check_worthiness` | `acc_norm` | 0.428702 |
| `relevance_judgment` | `acc` | 0.421389 |
| `relevance_judgment` | `acc_norm` | 0.558958 |
| `tquad` | `exact` | 0.000000 |
| `tquad` | `f1` | 5.260303 |

The [June 7 pipeline smoke test](../../artifacts/uhem_smoke_2026-06-07_job492393/metrics.json) is also retained: depth 2, 12,976,182 parameters, 259,522,560 training token positions, final/minimum validation BPB 1.239557, separate base-eval validation BPB 1.239951. It validates the training pipeline and is excluded from the controlled model comparison.

The repository has Atlas SFT preparation/launcher documentation, but no matched multi-tokenizer SFT result table in this evidence bundle. The d32 production work is a separate experiment and is excluded from the d20/d16/d12 ablation.

## Reproduce and interpret

Run `python -m scripts.build_model_study_results` from the repository root. The builder checks all 132 task cells, historical carry-forward consistency, final/minimum BPB equality, throughput arithmetic, and macro compatibility with four-decimal source rounding. It records SHA-256 hashes and preserves UHeM source paths in JSON.

Reported macros are preserved rather than recomputed as new measurements. In particular, the 64k TurkishDelight reported macro is 0.4567; the mean of its already-rounded task cells is 0.4567545. The difference is compatible with independent rounding of the source macro and task scores. Raw higher-precision results are needed to resolve its last decimal.

For a presentation: report the tokenizer diagnostics and archived BPB, label the CETVEL matrix as legacy, show the evaluator correction and missing 32k TurkishDelight model, and separate later model-scale/budget experiments. Tokenizer-effect claims require a corrected common evaluator, matched raw-text exposure/evaluation spans, replicated seeds, and paired statistical analysis.
