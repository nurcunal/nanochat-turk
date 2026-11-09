# Fresh matched tokenizer evaluation

UHeM CPU job **558060** completed on **9 November 2026** with exit code `0:0`
in **14m37s**. It evaluated all twelve archived tokenizer bundles on both train
and reserved-validation samples. No tokenizer or language model was retrained.

| Sample | Documents | Raw UTF-8 bytes | Regex words | Isolated words scored | Reference morphemes scored |
|---|---:|---:|---:|---:|---:|
| Train prefix | 50,000 | 186,328,421 | 23,042,031 | 500,000 | 768,779 |
| Reserved validation-shard prefix | 50,000 | 161,056,183 | 19,832,065 | 500,000 | 757,800 |

The training sample reproduces the original 50,000-document metrics: all twelve
tokenizers have exactly the archived document, byte, character, word, token and
morpheme-boundary crossing counts. The uniform 500,000-word cap replaces the
mixed 200,000/500,000-word caps in older word-level results. The validation
sample comes from the final source shard, excluded from the training split.
Both samples use **automatic TRmorph reference segmentation**, not independent
human morphology annotations.

The `train/` and `val/` folders each contain twelve complete metric JSON files
and a `requested_metrics.json` summary. New fields include exact morpheme-token
span preservation, isolated-word boundary precision/recall/F1, lossless
whole-word coverage, Unicode character-only segmentation, byte-only
segmentation, and applicable UNK counts. Isolated-word boundary recall differs
from contextual document-level boundary preservation. Character-only rates
include one-character words; a separate rate conditions on words longer than
one character.

All 24 evaluations scored exactly 500,000 words without excluded or
unalignable words. All 50,000 documents per evaluation round-tripped exactly.
UNK is not applicable to these byte-complete BPE models. Every consistency
evaluation used `sklearn_minibatch_kmeans`; none used the fallback clusterer.

- [manifest.json](manifest.json): authoritative output hashes, source snapshot,
  environment, sample provenance and matched-sample checks.
- [source_manifest.json](source_manifest.json): exact source and tokenizer file
  hashes. This was an isolated snapshot of the consolidation work in progress;
  per-file hashes identify the executed code, rather than the base Git revision
  alone.
- [executed-source.tar.gz](executed-source.tar.gz): exact executed Python/shell
  source and tokenizer configurations, verified against all 24 corresponding
  manifest hashes. Tokenizer binaries are not duplicated; their hashes match
  the twelve bundles already in `artifacts/tokenizers`. This also preserves
  the executed finalizer before a later idempotence-only maintenance fix.
- [sample_manifest.json](sample_manifest.json): source ordering, selected shards,
  sample counts and ordered raw/annotated document-content hashes.
- [environment.json](environment.json): Python and package versions.
- [job_accounting.json](job_accounting.json): Slurm completion and resources.

The run used Python 3.12.4, tiktoken 0.11.0, tokenizers 0.22.2, PyArrow 21.0.0,
PyTorch 2.9.1+cu128 and scikit-learn 1.9.0 on a CPU node. Four tokenizer
processes ran concurrently, with eight tokenizer threads each. Encoding speed
is therefore an execution diagnostic, not an isolated speed benchmark.

The isolated remote run directory is
`/ari/users/nunal/nanochat-tokenizer-study-20261109-v1`. The original source
repository and corpus were not modified. Reproduction helpers are
[`prepare_tokenizer_metric_sample.py`](../../../../scripts/prepare_tokenizer_metric_sample.py),
[`tokenizer_metrics.py`](../../../../scripts/tokenizer_metrics.py),
[`finalize_tokenizer_metric_study.py`](../../../../scripts/finalize_tokenizer_metric_study.py)
and [`uhem_tokenizer_metrics_study.sbatch`](../../../../runs/uhem_tokenizer_metrics_study.sbatch).
