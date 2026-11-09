# nanochat-turk

Turkish foundation-model training and a controlled morphology-aware tokenizer
study built on [karpathy/nanochat](https://github.com/karpathy/nanochat).

The project compares raw BPE with MorphBPE tokenizers trained from TRmorph,
Zemberek, and TurkishDelightNLP boundaries. FineWeb-2 Turkish data, training
token positions, optimizer settings, and evaluation are held fixed within each
vocabulary tier. The final tokenizers accept ordinary Turkish text and do not
need a morphological analyzer at inference.

Work continues on **`main`**. Historical branches are retained.
For presentation tables, provenance, and the full comparison, start with
[`docs/study_results/README.md`](docs/study_results/README.md).
The consolidated d32 production pipeline is documented in
[`docs/tr_d32_turkish_wsd_uhem.md`](docs/tr_d32_turkish_wsd_uhem.md);
its production work is separate from the completed tokenizer ablation below.

Start with the canonical
[`MorphBPE-alignment.md`](MorphBPE-alignment.md) for the paper comparison,
completed work, publication audit, limitations, and prioritized TODO list.

## Current Result

Across the currently checked outputs, all nine MorphBPE variants improve
Morphological Edit Distance and Morphological Consistency over same-size raw
BPE on the automatic TRmorph reference, while using more document tokens per
word. The
earlier 50k-document archive mixes 200k/500k word caps at 32k. The newly
recovered 93.21M-document corpus pass uses matched 500k word/alignment caps;
fresh train/validation diagnostics add explicit morpheme and coverage metrics.
The model table below preserves the historical measurements. A November UHeM audit recovered fixes
for CETVEL generation handling and continuation-boundary scoring; its
core-11 rankings are provisional, and the old XQuAD interpretation is
superseded. A corrected common eleven-model rerun is still needed.

| Tier | Raw BPE BPB | Best MorphBPE BPB | Raw legacy core-11 | Best MorphBPE legacy core-11 | Recorded observation |
| --- | ---: | ---: | ---: | ---: | --- |
| 32k / d20 | **0.6232** | 0.6250 | 0.4514 | **0.4618** | Raw BPE has lower archived BPB; legacy macro ranking needs reevaluation. |
| 64k / d16 | **0.6409** | 0.6510 | **0.4590** | 0.4568 | Raw BPE has lower archived BPB; legacy macro ranking needs reevaluation. |
| 128k / d12 | **0.6749** | 0.6820 | **0.4651** | 0.4618 | Raw BPE has lower archived BPB; legacy macro ranking needs reevaluation. |

Lower BPB is better. Core-11 is the mean of the common CETVEL
classification/log-likelihood tasks; XQuAD is reported separately because its
F1 scale is different. See the
[full CETVEL table](docs/cetvel_model_comparison.md) and the
[12-tokenizer intrinsic table](docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md).

MorphBPE improves the recorded morphology-sensitive tokenizer diagnostics
against an automatic TRmorph reference, while raw BPE has the lowest archived
BPB in every completed ablation tier. Fixed token budgets do not ensure equal
raw-text training/evaluation spans. The same d20 baseline's XQuAD F1 changes
from 3.0985 to 11.3369 after fixing generation handling, so the original QA
table cannot establish tokenizer benefits. See the
[evaluator audit and later models](docs/study_results/later_model_results.md),
including the separate d20 continuation and completed d32 FineWeb2 model.
The latter's 0.519040 BPB is a train-overlap diagnostic, not held-out loss.

## What Is Complete

- FineWeb-2 Turkish (`tur_Latn`) data download, deterministic manifests,
  checkpointing, UHeM/A100 training, and CETVEL evaluation.
- Exact-surface segmenter adapters for identity, TRmorph, Zemberek, and
  TurkishDelightNLP, with caching, fallback tracking, and resume-safe corpus
  segmentation.
- Paper-style raw-text MorphBPE training: boundaries constrain merge learning
  but never enter the language-model stream.
- All 12 tokenizers at 32k, 64k, and 128k.
- Eleven completed base models and a common CETVEL core-12 comparison. The only
  missing model cell is 32k/d20 TurkishDelightNLP.
- Fertility, Morphological Edit Distance, Morphological Consistency,
  boundary-crossing, compression, reversibility, and throughput metrics on a
  matched 50,000-document sample.
- Recovered full-corpus metrics for all twelve tokenizers: 93,211,827 training
  documents, verified job completion and source hashes.
- Fresh measurements for all twelve tokenizers on matched 50,000-document
  training and validation samples: morpheme preservation, boundary F1,
  isolated subword efficiency, whole-word and character coverage. UHeM CPU job
  `558060` completed successfully; all 24 outputs and source hashes are included.
- Compact tokenizers, manifests, metrics, and benchmark summaries checked into
  GitHub.

## Artifacts and Dated Publication Audit

| Artifact | GitHub | Hugging Face at the 2026-07-15 audit |
| --- | --- | --- |
| Source and compact documentation/results | This repository | Not applicable |
| 12 tokenizer bundles | [`artifacts/tokenizers`](artifacts/tokenizers/) | None is published standalone; raw BPE 32k is embedded in the public model and the other 11 are absent |
| Raw BPE 32k raw checkpoint | Summary and provenance here | [`nurcunal/nanochat-turk-d20-bpe32k`](https://huggingface.co/nurcunal/nanochat-turk-d20-bpe32k) |
| Other 10 completed models | Compact metrics and UHeM provenance here | Not uploaded |
| Eleven-model CETVEL comparison | [June 22 artifact](artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) | Not uploaded |

The authenticated publication audit was performed on 2026-07-15; the Hub column
is historical and was not re-audited for this results update. Empty Hub
IDs in tokenizer manifests are real release gaps, not evidence of private
repositories. Details and the exact missing-model inventory are in
[`MorphBPE-alignment.md`](MorphBPE-alignment.md#publication-audit).

## Method in This Repository

```text
raw FineWeb-2 Turkish
        |
        +--> raw BPE tokenizer ------------------------------+
        |                                                    |
        +--> morphological segmenter                         |
                  |                                          |
                  +--> boundary-marked training corpus       |
                              |                              |
                              +--> forbid cross-boundary      |
                                   BPE training merges        |
                                              |              |
                                              v              v
                                      standard raw-text tokenizers
                                              |
                                   matched base-model training
                                              |
                                  validation BPB + CETVEL
```

The main implementation is in
[`nanochat/morphology/morphbpe.py`](nanochat/morphology/morphbpe.py),
[`scripts/morph_segment_corpus.py`](scripts/morph_segment_corpus.py), and
[`scripts/tok_train.py`](scripts/tok_train.py). The contract and controls are
documented in
[`docs/tokenizer_tests/morphbpe_framework.md`](docs/tokenizer_tests/morphbpe_framework.md).

## Quick Start

Python 3.10 or newer and `uv` are expected.

```bash
uv sync --extra cpu --group dev
source .venv/bin/activate
pytest
```

Download a small Turkish data prefix and train the raw-BPE baseline tokenizer:

```bash
python -m nanochat.dataset -n 8
python -m scripts.tok_train \
  --implementation bpe \
  --tokenizer-name bpe_32768 \
  --vocab-size 32768
```

MorphBPE training requires a boundary-marked parquet corpus first. See the
[pipeline guide](docs/turkish_foundation.md),
[MorphBPE framework](docs/tokenizer_tests/morphbpe_framework.md), and
[run launcher index](runs/README.md) before launching segmentation or GPU jobs.

## Repository Guide

| Path | Purpose |
| --- | --- |
| [`MorphBPE-alignment.md`](MorphBPE-alignment.md) | Canonical paper alignment, evidence, release audit, and TODOs. |
| [`nanochat`](nanochat/) | Model, data, tokenizer, checkpoint, and evaluation code. |
| [`nanochat/morphology`](nanochat/morphology/) | Boundary logic, MorphBPE transform, and segmenter adapters. |
| [`scripts`](scripts/) | Training, metrics, evaluation, publication, and reporting commands. |
| [`runs`](runs/) | Local and UHeM/Slurm launchers; see its index before reuse. |
| [`docs`](docs/) | Stable study documentation and clearly marked historical operations notes. |
| [`artifacts`](artifacts/) | Compact tokenizer, checkpoint-smoke, CETVEL, and provenance artifacts. |
| [`tests`](tests/) | MorphBPE, segmenter, dataset, checkpoint, attention, and engine tests. |

The large [`docs/project_report_readme.md`](docs/project_report_readme.md) is a
historical project-memory log, not the current status page.

## Experimental Caveats

- The morphology metric reference is generated by TRmorph, which can favor the
  TRmorph-trained tokenizer. An independent annotated Turkish test set is still
  required.
- The 32k/64k/128k sweep is an engineering extension, not the MorphBPE paper's
  8k-increment vocabulary-selection procedure.
- Fixed token positions do not guarantee fixed raw bytes or documents when
  tokenizers have different fertility.
- CETVEL rows are single-run point estimates without paired significance tests.
- Cross-vocabulary results also change transformer depth; only comparisons
  inside one tier are tokenizer-controlled.

## Paper Reference

This work follows and extends:

> Ehsaneddin Asgari, Yassine El Kheir, MohammadAli SadraeiJavaheri, and Ali
> Nazari. 2026. [MorphBPE: Morphology-Aware Tokenization for Efficient LLM
> Training](https://aclanthology.org/2026.findings-acl.2068/). *Findings of ACL
> 2026*, 41610-41621.

The project uses [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
for Turkish pretraining data and CETVEL for Turkish model evaluation.

## License

MIT. This repository is a Turkish research fork of
[karpathy/nanochat](https://github.com/karpathy/nanochat); see
[`LICENSE`](LICENSE).
