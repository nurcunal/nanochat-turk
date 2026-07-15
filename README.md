# nanochat-turk

Turkish foundation-model training and a controlled morphology-aware tokenizer
study built on [karpathy/nanochat](https://github.com/karpathy/nanochat).

The project compares raw BPE with MorphBPE tokenizers trained from TRmorph,
Zemberek, and TurkishDelightNLP boundaries. FineWeb-2 Turkish data, training
token positions, optimizer settings, and evaluation are held fixed within each
vocabulary tier. The final tokenizers accept ordinary Turkish text and do not
need a morphological analyzer at inference.

Start with the canonical
[`MorphBPE-alignment.md`](MorphBPE-alignment.md) for the paper comparison,
completed work, publication audit, limitations, and prioritized TODO list.

## Current Result

Across the currently checked outputs, all nine MorphBPE variants improve
Morphological Edit Distance and Morphological Consistency over same-size raw
BPE on the automatic TRmorph reference, while using more tokens per word. The
32k outputs mix 200k- and 500k-occurrence metric caps, so this is a directional
pattern rather than a fully matched intrinsic estimate. Model results are mixed
rather than a universal MorphBPE win:

| Tier | Raw BPE BPB | Best MorphBPE BPB | Raw core-11 | Best MorphBPE core-11 | Main observation |
| --- | ---: | ---: | ---: | ---: | --- |
| 32k / d20 | **0.6232** | 0.6250 | 0.4514 | **0.4618** | Zemberek leads CETVEL macro; raw BPE leads BPB. |
| 64k / d16 | **0.6409** | 0.6510 | **0.4590** | 0.4568 | Raw BPE leads both aggregate metrics; MorphBPE leads XQuAD. |
| 128k / d12 | **0.6749** | 0.6820 | **0.4651** | 0.4618 | Raw BPE leads aggregate metrics; Zemberek leads XQuAD and Belebele. |

Lower BPB is better. Core-11 is the mean of the common CETVEL
classification/log-likelihood tasks; XQuAD is reported separately because its
F1 scale is different. See the
[full CETVEL table](docs/cetvel_model_comparison.md) and the
[12-tokenizer intrinsic table](docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md).

The appropriate claim is narrow: Turkish MorphBPE reliably improves
morphology-sensitive tokenizer diagnostics, but model benefits depend on
segmenter, vocabulary tier, and task. Raw BPE remains the best validation-BPB
choice in every completed tier.

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
- Compact tokenizers, manifests, metrics, and benchmark summaries checked into
  GitHub.

## Public Artifacts

| Artifact | GitHub | Hugging Face |
| --- | --- | --- |
| Source and compact documentation/results | This repository | Not applicable |
| 12 tokenizer bundles | [`artifacts/tokenizers`](artifacts/tokenizers/) | None is published standalone; raw BPE 32k is embedded in the public model and the other 11 are absent |
| Raw BPE 32k raw checkpoint | Summary and provenance here | [`nurcunal/nanochat-turk-d20-bpe32k`](https://huggingface.co/nurcunal/nanochat-turk-d20-bpe32k) |
| Other 10 completed models | Compact metrics and UHeM provenance here | Not uploaded |
| Eleven-model CETVEL comparison | [June 22 artifact](artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) | Not uploaded |

The authenticated publication audit was performed on 2026-07-15. Empty Hub
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
