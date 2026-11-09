# Additional LLM configurations and recovered benchmarks

These existing checkpoints and experiment outputs were recovered from UHeM on 9 November 2026. They extend the original [eleven-model tokenizer comparison](model_results.md) with continuation, model scaling, generation-protocol corrections and broader benchmarks. No new training or evaluation is claimed. Download [configuration CSV](model_configuration_results.csv), [complete JSON](model_configuration_results.json), [all available task metrics](model_all_task_metrics.csv), and [table-compatible protocol task scores](model_official_task_metrics.csv).

## Model configuration and training results

**The d32 BPB is a training-overlap diagnostic.** Its launch manifest makes all 30 source shards eligible for optimization and reuses one source shard for validation. It must not be presented as a held-out BPB improvement over d20 or MorphBPE. The completed FineWeb-2 d32 experiment is distinct from the separate general-corpus d32/WSD production plan.

| Experiment | Layers / width / heads | Parameters | Checkpoint | Batch positions | Cumulative positions | BPB ↓ | Training-loop h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d20 raw-BPE continuation | 20 / 1280 / 10 | 896,533,746 | 34,200 | 1,048,576 | 35,861,299,200 | 0.59145082 | 33.203 |
| d32 raw-BPE FineWeb-2 | 32 / 2048 / 16 | 2,818,575,450 | 28,000 | 2,097,152 | 58,720,256,000 | 0.51903967 | 120.750 |

Both use raw BPE with 32,768 entries, 2,048-token context and full attention. d20 continues the 17,100-step baseline to 34,200 cumulative steps with an explicitly changed continuation learning-rate schedule (start fraction 0.05, peak 0.25, 1,000-step warmup). Its BPB changes from archived 0.62321265 to 0.59145082; there is no recovered continuation CETVEL result. The checkpoint time counter is cumulative state and is not attributed solely to the continuation phase.

d32 uses 2,818,575,450 parameters, BF16, eight A100 GPUs, SDPA and 58,720,256,000 scheduled positions. Its 28,000 steps are a coverage-oriented budget; the legacy loader/resume behavior does not establish exactly-once exposure. Training-loop hours exclude initial compile/warmup steps, evaluation, checkpointing and launch overhead. Exact metadata and the original launch manifest are preserved in JSON.

### Measured performance and scheduler outcomes

**The d20 continuation saved its final step-34,200 checkpoint, but Slurm job 512018 ended FAILED (exit code 1).** Its final model measurements are available; it is not described as a cleanly completed job. The d32 job 512605 completed with exit code 0.

| Experiment | Job / state | Slurm wall time | Median throughput | Unit | Median step ms | BF16 MFU % | Peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d20 raw-BPE continuation | 512018 / FAILED (exit code 1) | 12:11:58 | 416325.5 | token positions/s | 2518.64 | 54.05 | 17847.49 |
| d32 raw-BPE FineWeb-2 | 512605 / COMPLETED (exit code 0) | 5-01:19:15 | 135456.5 | valid targets/s | 15482.08 | 63.37 | 51900.56 |

The medians use the final 100 printed iterations: d20 steps 34,100–34,199 and d32 updates 27,901–28,000. Both use eight A100s. The d20 logger measures token positions/s, whereas d32 measures valid prediction targets/s; preserve this distinction. The d20 job wall time is 12:11:58, while its inherited cumulative training-time counter is 33.203 hours. These durations cover different phases and must not be divided into the same token budget. Whole-log hashes and selected line numbers are retained in the evidence bundle.

## Legacy core-11: d20 and d32

The following preserves the original eleven classification/log-likelihood metrics. The legacy adapter did not validate the tokenized context/continuation boundary, so these values may be affected by likelihood-scoring errors. Their differences are provisional descriptive measurements, not established model-scale or tokenizer effects; a corrected common rerun is needed. The models also differ in scale, data eligibility and training budget despite sharing raw BPE.

| Task | Metric | d20 / step 17,100 | d32 / step 28,000 | Δ percentage points |
| --- | --- | --- | --- | --- |
| `exams_tr` | acc_norm | 0.3104 | 0.3410 | +3.05 |
| `belebele_tr` | acc_norm | 0.2522 | 0.2389 | -1.33 |
| `turkish_plu` | acc_norm | 0.5027 | 0.5504 | +4.77 |
| `cetvel_xcopa_tr` | acc | 0.6180 | 0.6900 | +7.20 |
| `cetvel_xnli_tr` | acc_norm | 0.3335 | 0.3329 | -0.06 |
| `mnli_tr` | acc_norm | 0.3210 | 0.3235 | +0.25 |
| `snli_tr` | acc_norm | 0.3234 | 0.3186 | -0.48 |
| `news_cat` | acc_norm | 0.6760 | 0.7040 | +2.80 |
| `offenseval_tr` | acc_norm | 0.7971 | 0.3370 | -46.00 |
| `trclaim19` | acc_norm | 0.4938 | 0.4835 | -1.03 |
| `xfact_tr` | acc_norm | 0.3373 | 0.2781 | -5.92 |
| **Core-11 mean** | mean | 0.4514 | 0.4180 | -3.34 |

## Generation protocol: corrected XQuAD

The legacy adapter misread generation kwargs, ignoring actual stop strings and task generation limits; its XQuAD values cannot support absolute QA or tokenizer-effect conclusions. The original scores are retained as historical evidence. Only the raw-BPE d20 reference and d32 have recovered corrected-generation runs in this bundle. They do not establish corrected scores for the other ten tokenizer-trained models.

| Model / protocol | Max generation tokens | Examples | Exact match % | F1 / 100 |
| --- | --- | --- | --- | --- |
| d20 baseline · legacy | 128 | 1190 | 0.0000 | 3.0985 |
| d20 baseline · corrected | 128 | 1190 | 4.5378 | 11.3369 |
| d32 · corrected | 128 | 1190 | 8.2353 | 15.8833 |
| d32 · Kumru comparison protocol | 256 | 1190 | 8.2353 | 15.1717 |

The corrected 128-token d20/d32 comparison gives F1 11.3369 versus 15.8833. The later 256-token d32 QA run gives 15.1717; both are retained with their own source paths and settings. Generation limits and the recorded protocol must accompany any score. The same-checkpoint d20 change is an evaluator correction, not a training gain; the d20/d32 contrast also changes model scale and training exposure. See [the evaluator audit](later_model_results.md) for the recovered bug mechanisms.

## Complete d32 CETVEL EACL table-compatible scoring

The saved receipt reports **37.1709/100**, the unweighted mean of seven category scores. The verification receipt covers **97,844 effective samples**, 24 top-level tasks and 38 verified leaves, with no sample limit. It records all eight source-shard hashes. This is the repository's `cetvel-eacl2026-table4-compatible-v1` protocol, not the legacy core-11 macro or an independent verification against an official public leaderboard.

Classification uses raw accuracy, QA normalized exact match, summarization TabiBERT BERTScore F1, translation COMET-22, and GEC stripped exact match. Do not substitute legacy `acc_norm`, ROUGE, BLEU or QA F1 into this aggregate. The receipt documents table-compatibility assumptions, model revisions and metric settings.

| Category | Score / 100 |
| --- | --- |
| QA | 8.3129 |
| MCQA | 39.9344 |
| TC | 38.3574 |
| NLI | 32.9338 |
| SUM | 81.0680 |
| MT | 58.6223 |
| GEC | 0.9678 |
| **Overall** | **37.1709** |

### All table-compatible protocol task scores

| Task | Metric | Examples | Score / 100 |
| --- | --- | --- | --- |
| `belebele_tr` | accuracy | 900 | 23.8889 |
| `bilmecebench` | accuracy | 442 | 21.7195 |
| `check_worthiness` | accuracy | 2188 | 37.6143 |
| `circumflex_tr` | accuracy | 70 | 48.5714 |
| `exams_tr` | accuracy | 393 | 31.2977 |
| `gecturk_generation` | stripped exact match | 20769 | 0.9678 |
| `ironytr` | accuracy | 600 | 50.0000 |
| `mkqa_tr` | SQuAD normalized exact match | 6758 | 4.4836 |
| `mlsum_tr` | BERTScore F1 | 12775 | 83.4049 |
| `mnli_tr` | accuracy | 10000 | 32.8700 |
| `news_cat` | accuracy | 250 | 70.0000 |
| `offenseval_tr` | accuracy | 3528 | 20.5215 |
| `relevance_judgment` | accuracy | 2188 | 59.5064 |
| `snli_tr` | accuracy | 10000 | 31.6600 |
| `sts_tr` | accuracy | 1379 | 12.9079 |
| `tquad` | SQuAD normalized exact match | 892 | 12.2197 |
| `tr-wikihow-summ` | BERTScore F1 | 5984 | 80.7533 |
| `turkce_atasozleri` | accuracy | 1730 | 0.0000 |
| `turkish_plu` | accuracy | aggregate | 51.2000 |
| `turkish_plu_goal_inference` | accuracy | 837 | 44.4444 |
| `turkish_plu_next_event_prediction` | accuracy | 655 | 49.0076 |
| `turkish_plu_step_inference` | accuracy | 612 | 39.3791 |
| `turkish_plu_step_ordering` | accuracy | 1021 | 65.2302 |
| `turkishmmlu` | accuracy | aggregate | 18.2222 |
| `wiki_lingua_tr` | BERTScore F1 | 874 | 79.7509 |
| `wmt-en-tr-prompt` | COMET-22 | 1001 | 62.1575 |
| `wmt-tr-en-prompt` | COMET-22 | 1001 | 55.0871 |
| `xcopa_tr` | accuracy | 500 | 69.0000 |
| `xlsum_tr` | BERTScore F1 | 3397 | 80.3629 |
| `xnli_tr` | accuracy | 5010 | 34.2715 |
| `xquad_tr` | SQuAD normalized exact match | 1190 | 8.2353 |

The MCQA definition includes both the Turkish-PLU aggregate and its listed component tasks, as specified by the archived table-compatible receipt. The translation task identifiers have counterintuitive direction names; use the receipt’s explicit source/target-language fields. The seven-category macro deliberately weights categories equally.

## Other recovered d32 metrics

The earlier Kumru-comparison output files also contain auxiliary metrics. These are separate measurements/aggregations from the table-compatible receipt above. Values retain their original scale. All auxiliary scores and available standard errors—including NLU subtasks—are included in the all-metric CSV and JSON.

| Task | Metric | Examples | Value | Scale |
| --- | --- | --- | --- | --- |
| `mkqa_tr` | em | 6758 | 0.044836 | native output |
| `mkqa_tr` | f1 | 6758 | 0.081235 | native output |
| `tquad` | exact | 892 | 12.219731 | 0–100 |
| `tquad` | f1 | 892 | 30.671475 | 0–100 |
| `xquad_tr` | exact | 1190 | 8.235294 | 0–100 |
| `xquad_tr` | f1 | 1190 | 15.171744 | 0–100 |
| `gecturk_generation` | exact_match | 20769 | 0.010641 | native output |
| `wmt-tr-en-prompt` | wer | 1001 | 12.494500 | native output |
| `wmt-tr-en-prompt` | bleu | 1001 | 0.015787 | native output |
| `mlsum_tr` | rouge1 | 12775 | 0.179007 | native output |
| `mlsum_tr` | rouge2 | 12775 | 0.108186 | native output |
| `mlsum_tr` | rougeL | 12775 | 0.153631 | native output |
| `mlsum_tr` | rougeLsum | 12775 | 0.147354 | native output |
| `tr-wikihow-summ` | rouge1 | 5984 | 0.151297 | native output |
| `tr-wikihow-summ` | rouge2 | 5984 | 0.048790 | native output |
| `tr-wikihow-summ` | rougeL | 5984 | 0.112428 | native output |
| `tr-wikihow-summ` | rougeLsum | 5984 | 0.112428 | native output |

The `kumru20_sum_fixed` file is a partial suite result containing complete outputs for the two listed summarization tasks; it is not a completed full-suite evaluation. The later table-compatible receipt includes four summarization tasks under BERTScore.

## Evidence coverage and unavailable results

| Evidence | Disposition |
| --- | --- |
| 11 original tokenizer-trained models | Archived scores retained; 132 primary cells checked against recovered full-precision outputs. |
| 32k TurkishDelight full d20 model | No completed result in the archived/recovered inventory. |
| d20 continuation step 34,200 | Checkpoint metadata present; no continuation benchmark found in this bundle. |
| d32 step 28,000 | Checkpoint, launch manifest, full table-compatible receipt, corrected QA and legacy/auxiliary evaluations present. |
| d32 smoke_job515374 / kumru20_comparison_smoke_job515380 | Two examples per task; retained in JSON/CSV with is_smoke=true; excluded from capability tables. |
| Raw d20 cetvel_out/full/cetvel_full_results.json | Recovery reports malformed JSON; not used. The valid archived partial/core comparison is retained. |
| Original raw d20 meta_017100.json | Absent at recovery; archived checkpoint statistics retained. |
| Later d32 SFT configuration screens | Reported in the separate [SFT screen report](sft_configuration_screens.md); these vary post-training settings on one raw-BPE model. |
| General-corpus d32 WSD | Separate production-plan documents; not the completed FineWeb-2 d32 model reported here. |
| June 7 d2 pipeline smoke | Separate archived infrastructure result; see model report. |

Rebuild all tables with `python -m scripts.build_model_study_results`. The builder verifies recovered primary task values, full-precision macros, checkpoint identity and receipt category/overall arithmetic. Source artifacts retain retrieval timestamps, UHeM paths, original source-file hashes, settings and reported errors.
