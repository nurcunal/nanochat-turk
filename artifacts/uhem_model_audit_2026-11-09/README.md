# UHeM model evidence recovered 2026-11-09

This read-only recovery retains compact metadata and benchmark metrics from
UHeM. Original large per-example payloads were omitted. Each source record
keeps its remote path, file size, timestamp and SHA-256 when available; the
local [manifest](manifest.json) hashes the extracted evidence files.

- `model-metadata-and-benchmarks.json`: all eleven original raw CETVEL result
  files and ten surviving step-17,100 metadata files. The raw-BPE 32k metadata
  is absent at its original path. All 132 published task scores match the
  recovered raw metrics at four decimals.
- `later-model-metadata-and-benchmarks.json`: d20 continuation and d32 FineWeb2
  metadata, August evaluator-fixed results, the complete EACL table-compatible
  scoring receipt, launch provenance, and explicitly recorded incomplete or
  smoke evidence.
- `evaluator_evidence/`: recovered generation/comparison runner source,
  launcher, and selected fixed-XQuAD log lines. These files document the
  historical correction; their presence does not install a runtime patch.
- `historical-training-reports.json` and `model-job-accounting.txt`: original
  training time, parameter/FLOP/memory statistics and scheduler completion.
  The raw-BPE32k report was overwritten by its later continuation, so original
  baseline cost metrics use its surviving job-492421 log excerpt instead.

The generation runner documents that the June adapter ignored task stop
strings and generation limits. The later comparison runner also fixes
continuation-boundary scoring. Original model rankings must therefore remain
qualified as legacy evaluator results until all tokenizer rows are reevaluated
under a common corrected protocol.

The completed d32 FineWeb2 model is a separate, non-WSD run from an August
content-addressed snapshot based on historical main. Its validation shard is
also trained on; its BPB is a train-overlap diagnostic. Smoke samples and a
malformed raw-d20 full-result file are not accepted as completed benchmarks.

See [model results](../../docs/study_results/model_results.md),
[later results and correction audit](../../docs/study_results/later_model_results.md),
and [reproduction notes](../../docs/study_results/reproduction.md).
