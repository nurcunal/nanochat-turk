# Pretraining and model benchmarks

**The tokenizer effect is task-dependent.** Raw BPE has the lowest archived validation BPB in all three tiers. Zemberek MorphBPE improves the 32k core-11 mean by **1.04 percentage points**; raw BPE leads that mean at 64k and 128k. MorphBPE improves XQuAD in selected tiers. These are single-run base-model results, not evidence of a universal MorphBPE advantage.

Historical tables below preserve results imported 2026-06-22; a 9 November 2026 recovery adds full-precision metrics, checkpoint settings and training performance. Eleven of twelve tokenizer-model cells have results; 32k TurkishDelight has no full model in this evidence. Additional continuation and d32 results are reported separately. No new evaluation is claimed.

Download [model summary CSV](model_results.csv), [all 132 task scores CSV](model_task_metrics.csv), or [full JSON with source paths and hashes](model_results.json). Original evidence: [June 22 benchmark artifact](../../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/metrics_summary.json).

## Experiment controls

Compare tokenizer families **within a vocabulary tier**. The documented design holds the raw FineWeb-2 Turkish source/manifest/order, model shape, optimizer recipe, token-position budget and evaluation harness fixed within that tier. Across tiers, depth changes to keep parameter counts near 0.9B.

| Vocab size | Transformer depth | Approx. parameters | Steps | Token positions/step | Total token positions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 20 | 896.5M | 17,100 | 1,048,576 | 17,930,649,600 |
| 65,536 | 16 | 872.4M | 17,100 | 1,048,576 | 17,930,649,600 |
| 131,072 | 12 | 890.2M | 17,100 | 1,048,576 | 17,930,649,600 |

The launch recipe uses four A100 GPUs and 2,048-token context. The recovered section below verifies effective configuration fields for ten final metadata records. The original raw-BPE d20 final metadata path was absent. Source initialization uses seed 42; repeated training seeds are not available. See the [design](../tokenizer_ablation_plan.md) and [launch record](../tokenizer_tests/vocab64_128_uhem_launch.md).

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

## Common CETVEL benchmark

Core-11 is the unweighted mean of eleven classification/log-likelihood task scores on a 0–1 scale. XQuAD F1 uses a 0–100 scale and is excluded from that mean. Core-11 deltas below are **percentage points**; XQuAD deltas are F1 points. Timing divides 39,441 expanded effective examples by the logged core-12 elapsed time. It is an end-to-end harness throughput proxy, not isolated tokenizer speed.

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

Legacy-generation XQuAD scores are low (about 2.25–3.48/100). A later corrected-generation raw-BPE d20 reference reaches 11.3369, reported separately below; the old scores should be interpreted with their legacy generation implementation. No confidence intervals or paired significance tests are available. The 128k TurkishDelight result file was complete at import time while Slurm was still finishing uploads; that archived job-state qualification is retained in JSON.

## Per-task results: 32k

Classification metrics below use 0–1; the final XQuAD row uses 0–100. Higher is better. `acc_norm` is the benchmark's length-normalized choice score; it is not F1.

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

## Per-task results: 64k

Classification metrics below use 0–1; the final XQuAD row uses 0–100. Higher is better. `acc_norm` is the benchmark's length-normalized choice score; it is not F1.

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

## Per-task results: 128k

Classification metrics below use 0–1; the final XQuAD row uses 0–100. Higher is better. `acc_norm` is the benchmark's length-normalized choice score; it is not F1.

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

The later [SFT configuration screens](sft_configuration_screens.md) compare post-training settings on the d32 raw-BPE model; they are separate from this matched tokenizer ablation. The completed FineWeb-2 d32 experiment and d20 continuation are reported in [additional model configurations](model_configuration_results.md), separately from this controlled tokenizer comparison and from the general-corpus d32/WSD production plan.

## Reproduce and interpret

Run `python -m scripts.build_model_study_results` from the repository root. The builder checks all 132 task cells, historical carry-forward consistency, final/minimum BPB equality, throughput arithmetic, and macro compatibility with four-decimal source rounding. It records SHA-256 hashes and preserves UHeM source paths in JSON.

Reported macros are preserved rather than recomputed as new measurements. In particular, the 64k TurkishDelight reported macro is 0.4567; the mean of its already-rounded task cells is 0.4567545. The difference is compatible with independent rounding of the source macro and task scores. The recovered full-precision task values reproduce 0.4567 when their mean is rounded, resolving the apparent discrepancy; they remain separate from the untouched historical task cells.

For a presentation: emphasize the within-tier trade-off, report BPB alongside the common benchmark, show the missing 32k TurkishDelight model explicitly, and treat small differences as descriptive. Strong causal claims require matching raw-text exposure and evaluation spans, replicated seeds, and paired statistical analysis.

## Recovered checkpoint configuration and training performance

The compact [UHeM evidence bundle](../../artifacts/model_evidence_2026-11-09/) was recovered on 9 November 2026. It supplies ten of the eleven final checkpoint metadata records and full-precision evaluation results for all eleven models. All **132 primary task cells** reproduce the archived four-decimal values; all recovered core-11 means reproduce their archived macros. The additional [all-metric CSV](model_all_task_metrics.csv) includes available auxiliary metrics and per-task standard errors, not just the original selected metric.

The table reports `loop_state.total_training_time`, an accumulated training-loop counter that excludes initial compile/warmup steps, evaluation, checkpoint and setup overhead. It is not scheduler wall-clock time. Throughput below uses logged step measurements; the full position budget is not divided by this partial-time counter. [CSV](model_training_performance.csv) and [JSON](model_results.json) retain exact values, model and optimizer settings, source paths and hashes.

| Tokenizer | Layers | Width | Heads | Recovered BPB ↓ | Training-loop h |
| --- | --- | --- | --- | --- | --- |
| bpe_32768 | 20 | — | — | — | — |
| morphbpe_trmorph_32768 | 20 | 1280 | 10 | 0.62657260 | 21.096 |
| morphbpe_zemberek_32768 | 20 | 1280 | 10 | 0.62497914 | 21.174 |
| bpe_65536 | 16 | 1024 | 8 | 0.64087624 | 15.123 |
| morphbpe_trmorph_65536 | 16 | 1024 | 8 | 0.65211758 | 15.100 |
| morphbpe_zemberek_65536 | 16 | 1024 | 8 | 0.65140315 | 15.097 |
| morphbpe_tdelight_65536 | 16 | 1024 | 8 | 0.65101966 | 15.058 |
| bpe_131072 | 12 | 768 | 6 | 0.67487410 | 11.061 |
| morphbpe_trmorph_131072 | 12 | 768 | 6 | 0.69171386 | 10.972 |
| morphbpe_zemberek_131072 | 12 | 768 | 6 | 0.69403741 | 10.970 |
| morphbpe_tdelight_131072 | 12 | 768 | 6 | 0.68201767 | 10.998 |

All ten recovered configurations use context 2,048, full attention (`L`), matching query/KV heads, global batch 1,048,576 token positions, per-device batch 4, and FP8 disabled. The original raw-BPE d20 final metadata path was absent at recovery; its archived BPB and existing benchmark remain valid historical evidence. A dash means the value is unavailable from this metadata source.

The [additional model-configuration report](model_configuration_results.md) contains the d20 continuation, completed d32 FineWeb-2 model, fixed-generation comparisons and full EACL table-compatible benchmark. It explicitly separates protocols and the d32 training-overlap BPB diagnostic from the controlled tokenizer ablation.


### Training throughput, memory and scheduler time

Fresh log recovery preserves all eleven completed runs. Tokens/s, step time and BF16 MFU are medians of printed steps 17,000–17,099 (100 steps per run); MFU follows the logger’s hardware/FLOP convention. Peak MiB is the logged training-process memory figure, not total cluster RAM. Scheduler wall time includes other job work and is distinct from the training-loop counter.

| Tokenizer | Parameters | Training min | Slurm wall time | Median tok/s | Median step ms | BF16 MFU % | Peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bpe_32768 | 896,533,746 | 1267.66 | 21:31:38 | 235775.0 | 4447.35 | 61.21 | 18499.62 |
| morphbpe_trmorph_32768 | 896,533,746 | 1265.77 | 21:30:53 | 236079.5 | 4441.61 | 61.29 | 18499.74 |
| morphbpe_zemberek_32768 | 896,533,746 | 1270.43 | 21:35:34 | 235298.0 | 4456.36 | 61.09 | 18499.62 |
| bpe_65536 | 872,416,058 | 907.39 | 15:30:31 | 329363.5 | 3183.64 | 53.13 | 14645.31 |
| morphbpe_trmorph_65536 | 872,416,058 | 905.97 | 15:28:04 | 329908.5 | 3178.38 | 53.22 | 14645.31 |
| morphbpe_zemberek_65536 | 872,416,058 | 905.82 | 15:27:54 | 329899.0 | 3178.47 | 53.22 | 14645.31 |
| morphbpe_tdelight_65536 | 872,416,058 | 903.50 | 15:25:56 | 331144.5 | 3166.52 | 53.42 | 14645.31 |
| bpe_131072 | 890,241,506 | 663.68 | 11:26:01 | 451859.5 | 2320.57 | 48.52 | 13066.91 |
| morphbpe_trmorph_131072 | 890,241,506 | 658.31 | 11:19:34 | 454632.5 | 2306.42 | 48.82 | 13065.79 |
| morphbpe_zemberek_131072 | 890,241,506 | 658.23 | 11:19:37 | 455047.0 | 2304.32 | 48.86 | 13068.04 |
| morphbpe_tdelight_131072 | 890,241,506 | 659.88 | 11:21:12 | 453685.0 | 2311.24 | 48.72 | 13066.91 |

### Standalone base-evaluation BPB from training logs

These logged train/validation evaluations run after the final checkpoint. They are a separate evaluation call and are preserved separately from checkpoint validation BPB; they must not silently replace the earlier measurement.

| Tokenizer | Checkpoint val BPB (log) | Base-eval train BPB | Base-eval val BPB |
| --- | --- | --- | --- |
| bpe_32768 | 0.623213 | 0.749119 | 0.623278 |
| morphbpe_trmorph_32768 | 0.626573 | 0.750972 | 0.626622 |
| morphbpe_zemberek_32768 | 0.624979 | 0.750180 | 0.625030 |
| bpe_65536 | 0.640876 | 0.779835 | 0.640925 |
| morphbpe_trmorph_65536 | 0.652118 | 0.778736 | 0.652161 |
| morphbpe_zemberek_65536 | 0.651403 | 0.778257 | 0.651429 |
| morphbpe_tdelight_65536 | 0.651020 | 0.776459 | 0.651062 |
| bpe_131072 | 0.674874 | 0.817503 | 0.674923 |
| morphbpe_trmorph_131072 | 0.691714 | 0.820881 | 0.691756 |
| morphbpe_zemberek_131072 | 0.694037 | 0.820229 | 0.694084 |
| morphbpe_tdelight_131072 | 0.682018 | 0.818073 | 0.682060 |

### Archived d2 pipeline smoke performance

This separate June 7 job validates the pipeline, not the controlled tokenizer comparison. Its tokenizer was trained on a 50-million-character cap and is named `bpe_32768_uhem_smoke`.

| Depth / width / heads | Context | Parameters | Steps | Training positions | Training min | Slurm wall time | MFU % | Peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 / 128 / 2 | 256 | 12,976,182 | 1980 | 259,522,560 | 11.60 | 00:17:38 | 2.51 | 1,420.94 |
The smoke artifact also records tokenizer-training time 6.3158 s, 7.347524e15 training FLOPs, 19.79 GB job memory utilized, and standalone train BPB 1.295173 / validation BPB 1.239951. Its checkpoint validation BPB is 1.239557.
