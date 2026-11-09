# Later model results and evaluator audit

**Recovered on 2026-11-09; measured in August 2026.** These results supplement the original tokenizer-ablation table. They change model depth, training budget and/or evaluation protocol, so they are not additional controlled tokenizer cells.

Sources: [compact UHeM audit](../../artifacts/uhem_model_audit_2026-11-09/), [derived JSON](later_model_results.json), and [runtime/reproduction notes](reproduction.md). Per-source paths, byte counts, timestamps and SHA-256 hashes are preserved. No model training or benchmark inference was rerun during recovery.

## A corrected evaluator changes the interpretation

The recovered [generation runner](../../artifacts/uhem_model_audit_2026-11-09/evaluator_evidence/cetvel_generation_fixed_runner.py) documents that the June adapter treated lm-eval's generation-kwargs mapping as a list of stop strings, ignoring actual `until` strings and task generation limits. The later [comparison runner](../../artifacts/uhem_model_audit_2026-11-09/evaluator_evidence/cetvel_comparison_runner.py) also moves trailing context whitespace to the continuation and checks that joint tokenization preserves an exact continuation boundary. Its code is archived as evidence; these files are not installed by the report builder.

The old eleven-model benchmark values remain visible as **legacy evaluator measurements**. The generation bug invalidates the previous absolute-XQuAD interpretation; the likelihood-boundary deviation can be tokenizer-dependent, so classification rankings also need a corrected common rerun. A single corrected raw-BPE row cannot repair the complete matrix.

| Model | Protocol | XQuAD F1 /100 | Exact match /100 |
| --- | --- | ---: | ---: |
| d20 BPE32k, step 17100 | June legacy adapter, cap 128 | 3.0985 | 0.0000 |
| d20 BPE32k, step 17100 | generation-fixed, CLI cap 128 | 11.3369 | 4.5378 |
| d32 BPE32k, step 28000 | generation-fixed, CLI cap 128 | 15.8833 | 8.2353 |
| d32 BPE32k, step 28000 | comparison runner, CLI cap 256 | 15.1717 | 8.2353 |

The two d32 F1 values use distinct recorded runner/generation settings and must not be mixed. The d20-versus-d32 rows also differ in scale and training exposure; they do not isolate a tokenizer effect. The fixed d20 run records the original `tr_d20_bpe_32768_chinchilla20` step-17,100 model in its evaluator arguments; its original checkpoint metadata path is now absent from UHeM, while the evaluation record remains.

## Additional pretraining

| Model | Depth | Step | Scheduled token positions | Final/minimum BPB | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `tr_d20_bpe_32768_cont40x` | 20 | 34,200 | 35,861,299,200 | 0.591451 | Continued d20 run; doubled budget |
| `tr_d32_bpe_32768_fineweb2_all_v1` | 32 | 28,000 | 58,720,256,000 | 0.519040 | Train-overlap diagnostic; not held out |

The d20 continuation uses a distinct learning-rate continuation policy and step 34,200; it must not replace the step-17,100 baseline in the original matrix.

The completed d32 model has 2,818,575,450 parameters, 2,048-token context, raw BPE32k and 58,720,256,000 scheduled positions. Its launch manifest explicitly trains on **all 30** FineWeb-2 Turkish source shards and also reuses the final source shard as validation. Consequently BPB **0.519040 is a train-overlap diagnostic**, not a held-out generalization result and not comparable as a held-out improvement over the original ablation. The launch describes the exposure schedule as a coverage-oriented heuristic, not a proof of exactly-once document/token consumption.

This completed run is `tr_d32_bpe_32768_fineweb2_all_v1`, deployed from a content-addressed source snapshot based on `955c3809c94c04731bf35a9695cc0a04c28e35e9` (snapshot SHA-256 `37503db09be2617d53d3f201d2fc776f7eb96844c7115818f02f38a3e0f83a97`). It records `strict_mode=false` and `lr_schedule=nanochat_linear`. It is **not** the frozen Turkish-general WSD v4 family whose source was subsequently consolidated into main.

The d32 legacy core-11 mean is **0.4180**, calculated from the same eleven selected metrics, but retains the legacy continuation-scoring caveat. It is not pooled into the tokenizer-ablation matrix.

## d32 CETVEL EACL table-compatible receipt

The recovered scoring receipt reports **37.1709/100**, the unweighted mean of seven category scores, and marks the result complete. This is the archived `cetvel-eacl2026-table4-compatible-v1` protocol result for d32 step 28,000; it is not the core-11 macro and is not an independent verification against a public leaderboard.

| Category | Score /100 |
| --- | ---: |
| QA | 8.3129 |
| MCQA | 39.9344 |
| TC | 38.3574 |
| NLI | 32.9338 |
| SUM | 81.0680 |
| MT | 58.6223 |
| GEC | 0.9678 |

The receipt pins CETVEL `6119c517e06cce23aaeac103ce3504c9380655e3`, lm-eval `6e49b1f6910931882a4b3b105794c6faf96b74e5`, and the metric provenance. Its compatibility warnings are retained: GECTurk uses stripped exact match; MCQA includes check-worthiness and relevance-judgment; summarization BERTScore uses pinned TabiBERT F1/final layer 22 with no baseline rescaling. These metrics differ from the legacy table's normalized accuracy, XQuAD F1 and ROUGE; do not compare their numerical scales directly.

All task scores in that receipt are preserved below. The receipt's QA values use exact match, not the F1 values in the correction table.

| Receipt task | Protocol score /100 |
| --- | ---: |
| `belebele_tr` | 23.8889 |
| `bilmecebench` | 21.7195 |
| `check_worthiness` | 37.6143 |
| `circumflex_tr` | 48.5714 |
| `exams_tr` | 31.2977 |
| `gecturk_generation` | 0.9678 |
| `ironytr` | 50.0000 |
| `mkqa_tr` | 4.4836 |
| `mlsum_tr` | 83.4049 |
| `mnli_tr` | 32.8700 |
| `news_cat` | 70.0000 |
| `offenseval_tr` | 20.5215 |
| `relevance_judgment` | 59.5064 |
| `snli_tr` | 31.6600 |
| `sts_tr` | 12.9079 |
| `tquad` | 12.2197 |
| `tr-wikihow-summ` | 80.7533 |
| `turkce_atasozleri` | 0.0000 |
| `turkish_plu` | 51.2000 |
| `turkish_plu_goal_inference` | 44.4444 |
| `turkish_plu_next_event_prediction` | 49.0076 |
| `turkish_plu_step_inference` | 39.3791 |
| `turkish_plu_step_ordering` | 65.2302 |
| `turkishmmlu` | 18.2222 |
| `wiki_lingua_tr` | 79.7509 |
| `wmt-en-tr-prompt` | 62.1575 |
| `wmt-tr-en-prompt` | 55.0871 |
| `xcopa_tr` | 69.0000 |
| `xlsum_tr` | 80.3629 |
| `xnli_tr` | 34.2715 |
| `xquad_tr` | 8.2353 |

Intermediate `kumru20_*` directories describe evaluation task suites, not a recovered Kumru model run: their model arguments identify the nanochat d32 checkpoint. Their full extracted metrics, including QA F1, GEC exact match, translation WER/BLEU and partial summarization ROUGE, remain in the source JSON; the table-compatible receipt uses its own documented aggregations.

## Excluded or incomplete evidence

Two-example smoke outputs are retained only as operations evidence. Their 50% XQuAD and 100% Belebele scores are not full benchmarks. The raw d20 `cetvel_out/full/cetvel_full_results.json` failed JSON parsing during recovery and is not accepted as a completed result. A `kumru20_sum_fixed` file is explicitly partial; it does not establish a completed suite by itself. The later complete table-compatible receipt is reported separately.

The original ten surviving step-17,100 metadata files contain effective user/model configuration but no complete training-code/environment fingerprint. All eleven original raw CETVEL result files survive and match the archived table at published precision. Their task standard errors are retained; they do not replace seed replication or paired uncertainty estimates for tokenizer differences.
