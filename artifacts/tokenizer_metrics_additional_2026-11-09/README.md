# Additional archived tokenizer measurements recovered from UHeM

Recovered on 9 November 2026 from `altay.uhem.itu.edu.tr`. All 28 JSON files are copied byte-for-byte from their source paths; `manifest.json` records original paths, sizes and SHA-256 hashes. The timestamped `slurm_accounting.txt` includes historical successful completion for jobs 496924 and 496925.

| Directory | Scope | Completion evidence |
|---|---|---|
| `sample_100k` | All 12 tokenizers, 100,000 training documents | Job 496924, COMPLETED, exit 0:0, 00:06:46 |
| `sample_1m` | All 12 tokenizers, 1,000,000 training documents | Job 496925, COMPLETED, exit 0:0, 00:24:03 |
| `d32_preflight` | Raw BPE32k tokenizer only, entire raw train and validation splits | Original complete preflight result JSONs; no matched MorphBPE validation rows |

The 100k and 1m passes use the same TRmorph-segmented training source as the earlier intrinsic study. Corpus metrics cover their stated document caps; isolated-word and morphology alignment still use 500,000 occurrences, and consistency uses 50,000 word types. These overlapping prefix samples are not independent replications.

The d32 preflight files name the original d20 raw BPE32k tokenizer path and evaluate raw text without a morphology reference. Their zero boundary totals mean not applicable. They are tokenizer diagnostics, not additional trained-model outcomes. The full raw train totals duplicate the corresponding full-corpus baseline totals.

Readable tables and complete machine-readable exports: [additional tokenizer report](../../docs/study_results/README.md#additional-tokenizers).
