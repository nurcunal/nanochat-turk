# dsai-3b-base benchmarks

**CETVEL EACL 2026 table-compatible score: 37.1709 / 100.** The archived
evaluation is complete, with 97,844 effective samples across 38 evaluated
leaves. This publication assigns the name `dsai-3b-base` to the existing
checkpoint; it does not represent a new training or evaluation run.

| Model identity | Value |
| --- | --- |
| Published name | `dsai-3b-base` |
| Original model tag | `tr_d32_bpe_32768_fineweb2_all_v1` |
| Checkpoint step | 28,000 |
| Parameters | 2,818,575,450 |
| Context length | 2,048 tokens |
| Tokenizer | Raw BPE, 32,768 vocabulary entries |
| Architecture | 32 layers, 2,048 embedding width, 16 attention heads |
| Checkpoint format | Raw nanochat PyTorch checkpoint |
| Checkpoint SHA-256 | `2123880d96b2de7d227ff31575b663ac291dcfd9fc9b9419086de032e58d0a98` |

The checkpoint was verified against this checksum on UHeM on 9 November 2026.
Model weights are not included in this GitHub artifact.

## CETVEL results

The score is the unweighted arithmetic mean of seven category scores.
The scoring receipt was created on 28 August 2026 (UTC), using protocol
`cetvel-eacl2026-table4-compatible-v1`.

| Category | Score / 100 |
| --- | ---: |
| QA | 8.3129 |
| MCQA | 39.9344 |
| TC | 38.3574 |
| NLI | 32.9338 |
| SUM | 81.0680 |
| MT | 58.6223 |
| GEC | 0.9678 |
| **Overall** | **37.1709** |

Full-precision category scores and all 31 reported task/group scores are in
[dsai-3b-base.json](dsai-3b-base.json) and [benchmarks.csv](benchmarks.csv).
Task/group scores include aggregates and should not be counted as 31
independent evaluation datasets.

## Interpretation and provenance

- This is an archived EACL table-compatible result, not an independently
  verified public-leaderboard entry.
- Classification uses raw accuracy, QA uses normalized exact match,
  summarization uses TabiBERT BERTScore F1, translation uses COMET-22, and
  GECTurk uses stripped exact match. The receipt preserves their exact
  settings, model revisions, sample counts, and compatibility warnings.
- The MCQA category includes check-worthiness and relevance judgment.
  Summarization pins TabiBERT layer 22 with no baseline rescaling. These
  details matter when comparing against other evaluation protocols.
- The saved BPB of **0.5190396722173233** is a training-overlap diagnostic:
  validation reuses a training shard. It is not held-out generalization loss.
- These results belong to the exact base-checkpoint hash above. They do not
  measure instruction-following or chat quality.

The following original evidence files are retained byte-for-byte:

- [Scoring receipt](evidence/cetvel_eacl2026_official_scoring_receipt.json):
  complete metrics, protocol revisions, metric definitions and source hashes.
- [Production verification](evidence/production_verification.json):
  complete production coverage with no sample limit.
- [Checkpoint metadata](evidence/meta_028000.json): model and training configuration.

[manifest.json](manifest.json) records the checksums of this publication.
The larger per-example prediction files remain referenced by their original
hashes in the scoring receipt. See the
[earlier model audit](../../docs/study_results/later_model_results.md) for
additional historical protocols, evaluator corrections and training lineage.
