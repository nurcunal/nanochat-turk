# Recovered UHeM model evidence — 9 November 2026

These compact JSON bundles preserve already-completed experiment outputs recovered from `altay.uhem.itu.edu.tr`. They contain checkpoint metadata, evaluation metrics (including auxiliary scores and available standard errors), evaluation settings, original source hashes and paths. The retrieval timestamps are in each bundle. They are evidence imports, not new training or evaluation runs.

| File | Coverage |
| --- | --- |
| [model-metadata-and-benchmarks.json](model-metadata-and-benchmarks.json) | All eleven original tokenizer-model benchmark outputs; ten final checkpoint metadata records; missing raw-d20 metadata explicitly recorded. |
| [later-model-metadata-and-benchmarks.json](later-model-metadata-and-benchmarks.json) | d20 continuation, completed d32 FineWeb-2 model, corrected-generation evaluations, auxiliary benchmarks, full official-scoring receipt and source verification. |
| [training-performance.json](training-performance.json) | Eleven original training logs: selected lines, whole-log hashes, last-100-step timing/MFU medians, parameter counts, memory and standalone BPB. |
| [later-training-performance.json](later-training-performance.json) | d20 continuation and d32 final training logs: measured rates, units, memory, cumulative counters and scheduler outcome; d20 saved its final checkpoint but its job ended FAILED. |
| [manifest.json](manifest.json) | Bundle hashes and import notes. |

The d32 launch manifest records training/validation overlap. Its BPB is a diagnostic on training-eligible data and must not be ranked against the original held-out validation results. Generation protocols also differ: retain the legacy, corrected-128-token, corrected-256-token and EACL table-compatible scores separately. Two-example smoke outputs and malformed/inaccessible source records are retained for completeness, not promoted to capability results.

Readable tables: [original model ablation](../../docs/study_results/README.md#original-models), [additional configurations and benchmarks](../../docs/study_results/README.md#model-configurations).
