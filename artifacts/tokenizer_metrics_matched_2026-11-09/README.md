# Completed matched tokenizer evaluation — 9 November 2026

UHeM job **558060** completed successfully (exit `0:0`, elapsed `00:14:37`). The final receipt records completion at `2026-11-09T12:57:09Z`. All 24 measurements are present: twelve tokenizers on each of the train and validation samples. The two `requested_metrics.json` files are aggregate summaries, not additional tokenizer treatments.

| Sample | Documents | UTF-8 bytes | Regex words | Word/morphology occurrence cap | Consistency word types |
|---|---:|---:|---:|---:|---:|
| train | 50,000 | 186,328,421 | 23,042,031 | 500,000 | 50,000 |
| val | 50,000 | 161,056,183 | 19,832,065 | 500,000 | 50,000 |

The sample-preparation manifest separates the first training prefix from the final reserved source shard. Reserved validation is this tokenizer evaluation's sample designation; the later d32 all-shard model training includes this text. The morphology reference is automatic TRmorph annotation, not independent human gold data.

This run adds exact individual-morpheme/token-span preservation, internal-boundary micro precision/recall/F1, Unicode-character-only word coverage, byte-only coverage and alignment-exclusion counts. It also remeasures the original fertility, compression, edit, consistency, roundtrip and timing diagnostics. Four concurrent tokenizer processes each used one metric worker and eight tokenizer threads. No model was retrained or benchmarked by this tokenizer job.

- `results/manifest.json`: finalizer receipt with per-file output hashes and identical word/reference sample hashes within each split.
- `source_manifest.json`: source snapshot identity and exact tokenizer/code hashes. The snapshot derives from in-progress work; its base Git revision does not imply the entire evaluated runtime equals that revision or current `main`.
- `source/`: exact frozen evaluator/support files, kept as evidence outside the active runtime. Their bytes match the source manifest. The twelve tokenizer bundles already checked in under `artifacts/tokenizers/` match the same manifest.
- `results/environment.json` and `results/sample_manifest.json`: package versions, selected shards, document totals, sampled-document hashes and reference corpus lineage.
- `results/train/` and `results/val/`: original per-tokenizer JSONs and logs, unchanged.
- `slurm_accounting.txt` and `tok-study-matched-558060.out/.err`: terminal scheduler and job-log evidence.
- `recovery_manifest.json`: hashes of every recovered file, including logs and frozen code.

The corpus and materialized sample parquet files remain on UHeM. [Readable report and downloads](../../docs/study_results/README.md#fresh-tokenizers). Rebuild tables from these completed outputs with `python -m scripts.build_fresh_tokenizer_results`; rebuilding does not rerun the evaluator.
