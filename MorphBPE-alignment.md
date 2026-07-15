# MorphBPE Alignment, Evidence, Publication Audit, and TODOs

Last verified: **2026-07-15**

This is the canonical status document for the Turkish MorphBPE work in
`nanochat-turk`. It compares the repository with the peer-reviewed MorphBPE
paper, separates completed evidence from planned work, and records what is and
is not publicly released. Older launch and project-memory notes are historical
provenance and should not override this file.

## Executive Assessment

The project implements the paper's **central tokenizer-training idea**:
morphological segmentation constrains BPE merge learning, the saved tokenizer
processes raw Turkish, and no segmenter is needed during language-model training
or inference. On the current automatic, in-domain TRmorph reference, every
Turkish MorphBPE variant also matches the paper's directional intrinsic pattern:
better morphological edit distance and consistency than same-size raw BPE, at
the cost of higher fertility.

The model evidence is more mixed than the paper's headline result:

- Logged final token-level training loss is lower for the completed MorphBPE
  models, which points in the same direction as the paper's cross-entropy
  curves.
- Validation bits per byte (BPB), the repository's tokenizer-normalized model
  metric, is better for raw BPE in every completed 32k, 64k, and 128k tier.
- MorphBPE wins selected CETVEL slices, especially at 32k and on XQuAD in
  several tiers, but raw BPE has the best core-11 macro at 64k and 128k.
- On the paper's own downstream benchmark family, Turkish Belebele, MorphBPE is
  below raw BPE at 32k and 64k; the 128k Zemberek model has the largest positive
  point estimate. No paired significance analysis has been run.

The defensible conclusion is therefore: **the current implementation behaves
like MorphBPE on the repository's intrinsic diagnostic, but it does not yet
provide independent-gold validation or establish a universal language-model or
downstream advantage over raw BPE.**

## Canonical Paper

Primary reference:

> Ehsaneddin Asgari, Yassine El Kheir, MohammadAli SadraeiJavaheri, and Ali
> Nazari. 2026. "MorphBPE: Morphology-Aware Tokenization for Efficient LLM
> Training." *Findings of ACL 2026*, pages 41610-41621.
> [ACL Anthology](https://aclanthology.org/2026.findings-acl.2068/) -
> [PDF](https://aclanthology.org/2026.findings-acl.2068.pdf) -
> [DOI](https://doi.org/10.18653/v1/2026.findings-acl.2068)

The repository's older notes cited
[arXiv:2502.00894 v1](https://arxiv.org/abs/2502.00894). The ACL version is now
canonical: it adds Ali Nazari, SentencePiece Unigram comparisons, zero-shot
Belebele evaluation with significance testing, full-text and token-length
checks, and expanded limitations. The authors' current implementation is
[qcri/MorphBPE](https://github.com/qcri/MorphBPE).

## What the Paper Does

### Method

MorphBPE changes only tokenizer training (paper Section 3, Algorithm 1):

1. Initialize BPE with individual characters.
2. obtain morpheme boundaries for the tokenizer-training material;
3. compute adjacent-symbol frequencies as in standard BPE;
4. select the most frequent pair whose symbols are inside the same morpheme;
5. skip a cross-boundary pair and take the next valid candidate;
6. repeat to the target vocabulary size.

The learned artifact is a standard deterministic BPE tokenizer. Boundaries,
analyzers, lexicons, and lookup tables are not required at inference. This is a
training-time bias, not a runtime guarantee: a merge learned inside one
morpheme can still cross the true boundary of a different unseen word.

The paper describes a character-initialized BPE. This repository implements
the constraint through nanochat's byte-level `rustbpe`/`tiktoken` stack. That
preserves arbitrary UTF-8 text and raw-text inference, but it is an engineering
adaptation rather than a byte-for-byte reproduction of Algorithm 1.

### Experimental Design

| Paper component | Camera-ready design |
| --- | --- |
| Languages | English, Russian, Hungarian, Arabic |
| Morphology data | SIGMORPHON 2022 plus Arabic resources; manually annotated data split 80/10/10 |
| Automatic morphology | One million Farasa Arabic forms used for tokenizer training only, excluded from intrinsic evaluation |
| LM data | Matched monolingual FineWeb2 subsets |
| Vocabulary choice | 8k-96k sweep in 8k increments; choose the first development-set `mu_e` plateau using a paired t-test |
| Selected vocabularies | Hungarian 24k, Russian 64k, English 96k, Arabic 96k |
| Tokenizer baselines | Standard BPE throughout; SentencePiece Unigram only as a tokenizer-level `mu_c` comparator in Table 2 |
| Models | Decoder-only 300M and 1B models |
| Training budget | Approximately 6B tokens at 300M and 20B tokens at 1B |
| Controls | Same text, vocabulary size, architecture, batch size, schedule, seed, and optimizer within each pair |
| Intrinsic tokenizer metrics | Fertility `phi`, raw Morphological Edit Distance `mu_e`, Morphological Consistency precision/recall/F1 `mu_c` |
| Consistency sampling | `k=100` clusters, `C=50` pairs per cluster, `N=10` bootstrap resamples |
| LM metric | Token-level cross-entropy, compared only at an identical vocabulary size |
| Downstream metric | Zero-shot Belebele accuracy from conditional answer log-probabilities |
| Significance | Paired McNemar test, `10^6` bootstrap resamples, Benjamini-Hochberg FDR at `alpha=0.05` |

### Paper Findings and Boundaries

The paper reports lower `mu_e`, higher `mu_c`, marginally higher fertility, and
lower token-level training cross-entropy for MorphBPE across its four
languages. On Belebele, the average improvement is about one percentage point;
Arabic and Russian are significant, Hungarian is not, and English is
negligible. This is a targeted result for productive morphology, not a uniform
downstream win.

The main experiments train tokenizers on word lists. Appendix A's only
full-running-text check compares raw BPE with Farasa-presegmented MorphBPE on
Arabic Wikipedia; the configuration detail is insufficient for an exact
reproduction. Appendix B argues that token length does not explain the loss
improvement, but it excludes Arabic and uses 16k vocabularies rather than the
main 24k/64k/96k settings. Its prose describes differences of 0.05-0.14
characters while the displayed frequency-weighted differences span 0.04-0.22.
The paper otherwise covers only monolingual models, two model scales, and one
downstream benchmark, and it depends on reliable morphological supervision.

There is also a small camera-ready inconsistency: Table 2 gives Hungarian
MorphBPE and Unigram the same `mu_c` F1 of 0.87, while the prose says MorphBPE
is higher for all four languages.

## Project-to-Paper Alignment

| Area | Repository implementation/evidence | Status |
| --- | --- | --- |
| Training-time boundary constraint | `nanochat/morphology/morphbpe.py` strips `U+E000` markers and emits BPE training chunks that cannot cross marked boundaries. | Aligned |
| Base alphabet | The paper initializes characters; the repository uses byte-level `rustbpe`/`tiktoken`. | Deliberate implementation divergence |
| Standard raw-text inference | `scripts/tok_train.py` saves `morphbpe` tokenizers with no decode strip and no runtime segmenter. Round-trip tests cover Turkish text. | Aligned |
| Exact reconstruction | Segmenter adapters accept a split only when surface pieces reconstruct the original word; otherwise they fall back or fail in strict mode. | Aligned and strengthened |
| Full-text tokenizer training | TRmorph, Zemberek, and TurkishDelightNLP segment FineWeb-2 Turkish running text before merge training. | Practical extension |
| Fertility, `mu_e`, `mu_c` | `scripts/tokenizer_metrics.py` implements all three and the paper defaults `k=100`, `C=50`, `N=10`; parts of the paper's `mu_c` representation and sampling procedure are underspecified. | Paper-compatible; exact match unverified |
| Metric dispersion | Precision/recall/F1 standard deviations are stored in JSON. | Aligned |
| Gold morphology test data | Current paper-style metrics use a TRmorph-segmented FineWeb-2 sample rather than an independent annotated Turkish test split. | Major gap |
| Vocabulary selection | Repository tests 32k/64k/128k engineering tiers, not the paper's 8k-step development-set stopping rule. | Deliberate extension; reproduction gap |
| Same-vocabulary BPE controls | Each completed tier has a raw-BPE baseline and fixed architecture within that tier, but fixed token budgets do not guarantee matched raw-text exposure. | Aligned on vocabulary and architecture; exposure controls incomplete |
| Unigram `mu_c` comparator | No same-corpus SentencePiece Unigram tokenizer comparison is present. The paper does not train a Unigram LM. | Gap relative to final paper |
| Model scale/budget | 32k/d20, 64k/d16, and 128k/d12 are about 872M-897M parameters and use 17.93B token positions. | Near-scale Turkish extension; no 300M arm |
| LM comparison | Final train loss, validation BPB, and CETVEL are recorded. Full matched loss curves and replicate statistics are not checked in. | Partial |
| Downstream benchmark | CETVEL includes `belebele_tr` and 10 other macro tasks, plus XQuAD. | Broader Turkish extension |
| Downstream significance | Only point estimates are archived; paired predictions, confidence intervals, McNemar/bootstrap tests, and FDR correction are absent. | Gap |
| Public release | Code and compact artifacts are on GitHub; most models, tokenizers, and later results are not on Hugging Face. | Incomplete |

## What Has Been Completed

### Pipeline and Implementation

- FineWeb-2 Turkish (`tur_Latn`) download, deterministic parquet ordering,
  validation-shard handling, and UHeM/A100 training paths.
- Boundary representation and raw-text MorphBPE merge training.
- Resume-safe, cached corpus segmentation with exact-surface validation.
- Four segmenter/control backends: identity, TRmorph, Zemberek, and
  TurkishDelightNLP.
- A deterministic hash-100k segmenter screen and a blind 500-item local judge.
- Checkpoint metadata that binds a model to its tokenizer artifact.
- Paper-style tokenizer metrics plus compression, boundary-crossing,
  reversibility, vocabulary-shape, and throughput diagnostics.
- Unit coverage for merge constraints, metrics, segmenter parsing/fallback,
  checkpoint selection, dataset columns, and attention/engine behavior.

### Produced Tokenizers and Models

- All 12 tokenizer cells exist: raw BPE, TRmorph MorphBPE, Zemberek MorphBPE,
  and TurkishDelightNLP MorphBPE at 32k, 64k, and 128k.
- Eleven full base models reached step 17,100. The only missing model cell is
  32k/d20 TurkishDelightNLP MorphBPE.
- Eleven completed models have common CETVEL core-12 summaries. Compact
  metrics and per-task values are in
  [`artifacts/cetvel_core12_tokenizer_ablation_2026-06-22`](artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/).
- The checked-in intrinsic comparison covers the same first 50,000 documents
  for all 12 tokenizers and uses raw text for encoding.

## Current Evidence

### Tokenizer-Level Result

The table summarizes the matched-vocabulary pattern from
[`tokenizer_metrics_comparison.json`](docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.json).
"Morph range" covers TRmorph, Zemberek, and TurkishDelightNLP; "best" chooses
the best MorphBPE value for that metric, not one synthetic winner.

| Vocab | Raw `phi` | Morph `phi` range | Raw `mu_e` | Best Morph `mu_e` | Raw `mu_c` | Best Morph `mu_c` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32k | 1.6157 | 1.7366-1.8166 | 1.6836 | 1.4126 | 0.3241 | 0.5129 |
| 64k | 1.4856 | 1.6091-1.7121 | 1.4627 | 1.1697 | 0.2863 | 0.4421 |
| 128k | 1.3915 | 1.5109-1.6304 | 1.2623 | 1.0045 | 0.2412 | 0.3971 |

Across the currently checked outputs, all nine MorphBPE rows improve both
morphology metrics over same-size raw BPE, and all nine spend more tokens per
word. This matches the paper's directional intrinsic trade-off on the current
automatic, in-domain reference. It is not yet a fully matched estimate: the 32k
raw/TRmorph/Zemberek files use a 200,000-occurrence cap while the 32k
TurkishDelight and other tiers use 500,000. The absolute values are not directly
comparable because the language, morphology references, corpora, and samples
differ, and parts of the paper's `mu_c` procedure are underspecified.

TRmorph supplies the evaluation reference for every row, but it supplies
tokenizer-training boundaries only for the TRmorph MorphBPE rows. In addition,
the 50,000-document sample comes from the FineWeb-2 corpus used to train the
tokenizers rather than a held-out intrinsic test. This setup can particularly
favor the TRmorph-trained tokenizer and makes every row dependent on one
automatic analyzer's notion of morphology; an independent Turkish gold set is
required.

### Model and Downstream Result

| Tier | Best BPB | Final train loss: raw / best Morph | Core-11 macro: raw / best Morph | Belebele: raw / best Morph | XQuAD F1: raw / best Morph |
| --- | --- | ---: | ---: | ---: | ---: |
| 32k/d20 | raw 0.6232 | 2.4899 / 2.0106 | 0.4514 / **0.4618** | **0.2522** / 0.2433 | 3.0985 / **3.4786** |
| 64k/d16 | raw 0.6409 | 2.5812 / 2.2754 | **0.4590** / 0.4568 | **0.2433** / 0.2356 | 2.8576 / **3.3280** |
| 128k/d12 | raw 0.6749 | 3.0976 / 2.5947 | **0.4651** / 0.4618 | 0.2389 / **0.2633** | 2.2674 / **2.9685** |

"Best Morph" may name a different segmenter in each metric. These are
descriptive selections, not a composite rank or significance-tested model
choice.

The lower logged training loss resembles the paper's finding, but it is not yet
a clean reproduction. MorphBPE's Turkish fertility is materially higher, the
repository has final logged losses rather than complete controlled curves, and
token-level loss can reward a finer tokenization. Validation BPB normalizes NLL
by raw bytes and favors raw BPE in every completed tier; it does not correct the
unequal training-text exposure that a fixed token budget can create. A
paper-strength result needs matched curves, token-length controls, exposure
accounting, and repeated seeds.

### Claim Status

| Candidate claim | Assessment |
| --- | --- |
| Morphology can constrain BPE training without runtime segmentation. | Supported by implementation and tests. |
| Turkish MorphBPE improves morphological alignment and consistency over same-size BPE. | Supported on the current TRmorph-referenced 50k-document sample. |
| Turkish MorphBPE lowers token-level training loss. | Observed at the final logged point, but granularity and curve controls remain. |
| Turkish MorphBPE improves validation language modeling. | Not supported by BPB; raw BPE wins every completed tier. |
| Turkish MorphBPE improves aggregate CETVEL. | Supported only for the 32k Zemberek core macro; not at 64k or 128k. |
| Turkish MorphBPE improves Belebele. | Mixed: negative at 32k/64k, positive for 128k Zemberek, no significance test. |
| One segmenter/tokenizer is the universal winner. | Not supported. The winner changes by metric and tier. |

## Main Validity and Reproducibility Gaps

1. **No independent gold Turkish morphology test split.** Automatic TRmorph is
   the reference for every tokenizer and the training supervision for the
   TRmorph MorphBPE rows. The 50k-document diagnostic is also sampled from the
   tokenizer-training corpus rather than held out.
2. **No paper-style vocabulary selection.** Cross-tier results also change
   transformer depth, so they are a budget-allocation study rather than a pure
   tokenizer comparison.
3. **No Unigram `mu_c` comparator.** The final ACL paper compares a
   SentencePiece Unigram tokenizer in its consistency table; it is not an LM or
   downstream baseline.
4. **Fixed token positions are not fixed text exposure.** Higher fertility can
   change raw bytes and documents consumed. Those exposure counts are not
   archived for every model.
5. **No replicate/significance analysis.** CETVEL values are single-run point
   estimates; paired per-example outputs and confidence intervals are not
   published.
6. **Incomplete dependency provenance.** FineWeb manifests often point at
   mutable `main`; segmenter/model revisions are not consistently pinned.
7. **Metric environment can change.** `scikit-learn` is not declared in
   `pyproject.toml`; without it, `mu_c` silently uses a deterministic hash
   fallback instead of MiniBatchKMeans.
8. **Generated artifacts can drift.** Existing publish manifests have blank
   Hub IDs and stale README hashes, and several omit Git commit/branch data.
9. **Full-corpus metrics are unverified.** Job `496882` was submitted, but no
   completed result is present in the checkout.
10. **The 32k TurkishDelight model cell is missing.** Its tokenizer exists, but
    no completed d20 checkpoint is documented.
11. **The base alphabet differs from the paper.** Byte-level merge training has
    not been isolated against a character-initialized implementation, so the
    effect of this engineering choice is unknown.

## Publication Audit

The following was verified against public GitHub state, authenticated Hugging
Face owner inventory, repository manifests, and file hashes on 2026-07-15.

### GitHub

- [`nurcunal/nanochat-turk`](https://github.com/nurcunal/nanochat-turk) is
  public and MIT licensed.
- At audit start, GitHub's default branch was `master` at `b5171c5`; the
  complete Turkish work was on `nanochat-turkish` at `b7e78a4`, exactly 97
  commits ahead. A `main` branch did not yet exist.
- The Turkish branch contains all 12 tokenizer bundles, the 11-model compact
  CETVEL artifact, paper-style tokenizer metrics, and judge summaries.
- There are no tags, GitHub Releases, open pull requests, or branch protection.
- Large smoke checkpoints and a CETVEL archive were committed directly. Future
  weights and raw outputs should use Hugging Face, Releases, or LFS; rewriting
  published history is not recommended as routine cleanup.

The target of this maintenance change is to create and push the requested
`main` line from the completed Turkish work. GitHub's default branch must then
be changed separately so this README becomes the repository landing page.

### Hugging Face

Only one project model is present in the authenticated owner inventory:

- [`nurcunal/nanochat-turk-d20-bpe32k`](https://huggingface.co/nurcunal/nanochat-turk-d20-bpe32k)
  is public, ungated, and about 6.66 GB. It contains the raw 32k model,
  optimizer shards, tokenizer, reports, and the raw-model CETVEL tasks 01-13
  subset. It is a nanochat training/resume checkpoint built from raw `.pt`
  files, not a Transformers `from_pretrained` model. The card is stale: it says
  CETVEL is absent or unresolved and does not identify the source branch and
  commit even though evaluation outputs are present. The three embedded
  tokenizer files plus `manifest.json` and `metrics_summary.json` from the
  compact CETVEL subset match their local copies byte for byte; the rest of the
  remote payload was not checksum-compared.

Missing from Hugging Face:

- the intended `nurcunal/nanochat-turk-tokenizers` repository (authenticated
  owner lookup returns not found);
- all 12 standalone tokenizer bundles in the intended release layout (raw BPE
  32k is embedded in the existing model repository; the other 11 are absent);
- 10 completed models: 32k TRmorph and Zemberek; all four 64k models; and all
  four 128k models;
- the later 11-model CETVEL comparison, the 12-tokenizer metric comparison,
  and a dedicated result/dataset repository.

Nine tokenizer publish manifests have an empty `repo_id`; the three original
32k bundles—`bpe_32768`, `morphbpe_trmorph_32768`, and
`morphbpe_zemberek_32768`—lack a publish manifest. Within the local artifact
tree, 52 recorded payload hashes validate, but all nine recorded README hashes
are stale because the cards were edited after manifest creation. Regenerate
manifests before uploading.

### Suitable Release Layout

| Site | Keep there |
| --- | --- |
| GitHub | Source, small configs/manifests, compact JSON/Markdown summaries, tests, launchers, documentation |
| Hugging Face model repos | Inference-oriented weights/tokenizers and, separately if needed, resumable raw checkpoints |
| Hugging Face tokenizer repo or Collection | All 12 tokenizer bundles with consistent cards, revisions, checksums, and loading instructions |
| Hugging Face dataset repo | CETVEL summaries/predictions where licenses allow, tokenizer metrics, data lineage, and evaluation provenance |
| UHeM/private storage | Licensed/restricted raw data, caches, temporary segmented corpora, and intermediate checkpoints |

## Prioritized TODO List

### P0 - Make the Existing Study Releasable

- [x] Create one canonical, camera-ready-paper alignment document.
- [x] Audit the full tracked repository, GitHub branches, authenticated Hugging
  Face inventory, artifact hashes, and stale documentation.
- [x] Make the root README a concise current landing page and add repository
  indexes for documentation and launchers.
- [ ] Push the completed work to `main` and make `main` the GitHub default
  branch.
- [ ] Regenerate all 12 tokenizer bundles' manifests after final card edits;
  populate Hub repo ID, source commit, immutable dataset revision, segmenter
  revision, command, environment, and checksums.
- [ ] Create `nurcunal/nanochat-turk-tokenizers` (or a documented Collection)
  and publish all 12 bundles.
- [ ] Refresh the existing raw-BPE model card: exact Git commit, current CETVEL
  results, limitations, raw-checkpoint loading contract, and links to code and
  result artifacts.
- [ ] Publish the 10 missing completed model cells. Keep optimizer shards out of
  inference-oriented repos unless resumable training is an explicit goal.
- [ ] Create a Hugging Face dataset/result repo for the 11-model CETVEL summary,
  12-tokenizer metrics, paired predictions where licensing permits, and
  provenance.
- [ ] Tag a reproducible GitHub release after Hub uploads and link every model,
  tokenizer, result artifact, paper citation, and checksum from the release.

### P0 - Repair Reproducibility Before New Claims

- [ ] Freeze an independently annotated or manually verified Turkish morphology
  train/dev/test resource; never score on automatic analyses used as training
  supervision.
- [ ] Pin FineWeb-2 to an immutable revision and record the exact ordered shard
  manifest for every tokenizer and model.
- [ ] Pin TRmorph FST, Zemberek JAR/model, TurkishDelight weights/code, CETVEL,
  and metric-library versions.
- [ ] Declare `scikit-learn` for paper metrics or remove the fallback from
  paper-facing runs; record the clustering implementation in every result.
- [ ] Archive seed, hyperparameters, optimizer/schedule, hardware, raw bytes,
  documents, and tokens consumed for every matched pair.
- [ ] Sanitize public logs and raw benchmark payloads for private paths, ANSI
  control codes, data redistribution terms, and unnecessary examples.

### P1 - Close the Paper-Reproduction Gaps

- [ ] Run a Turkish 8k-96k vocabulary sweep in 8k increments on the frozen dev
  set and apply the paper's paired `mu_e` plateau test.
- [ ] Train a same-corpus SentencePiece Unigram tokenizer at the selected
  Turkish vocabulary size and include it in the `mu_c` comparison; do not imply
  that the paper trained a Unigram language model.
- [ ] Complete the 32k/d20 TurkishDelight model or explicitly remove that cell
  from the preregistered matrix with a documented reason.
- [ ] Export matched BPE/MorphBPE loss curves and compare only identical
  vocabulary/architecture/data/seed settings.
- [ ] Add mean and frequency-weighted token-length tables plus fixed-byte and
  fixed-document robustness controls.
- [ ] Compare the current byte-level implementation with a
  character-initialized MorphBPE control to quantify the Algorithm 1
  adaptation.
- [ ] Retain paired Turkish Belebele predictions and run McNemar, bootstrap
  confidence intervals, and multiple-comparison correction.
- [ ] Repeat promoted pairs across multiple seeds; report effect sizes and
  uncertainty instead of best-row point estimates.
- [ ] If compute permits, add the paper's 300M/approximately 6B arm; otherwise
  label this project a Turkish extension of the 1B arm.

### P2 - Repository and Research Maintenance

- [ ] Verify UHeM job `496882`; import completed full-corpus outputs or record
  the terminal failure and stop referring to it as active.
- [ ] Make metric-table generation deterministic, including compact display
  labels, and add a CI drift check for JSON/Markdown pairs.
- [ ] Add lightweight CI for unit tests, Python compilation, JSON/TOML parsing,
  shell syntax, and internal Markdown links.
- [ ] Separate CETVEL into a locked environment so its older `datasets`
  requirement cannot mutate the training environment.
- [ ] Consolidate overlapping Slurm launchers into parameterized entry points;
  retain dated scripts only when they carry unique provenance.
- [ ] Add a small expert-reviewed Turkish segmentation error analysis covering
  ambiguity, derivation/inflection, apostrophes, named entities, noise, and
  analyzer fallback.

## Definition of Done

The project is paper-complete when a reader can clone a tagged commit, obtain
immutable data/segmenter references, rebuild the selected BPE/MorphBPE
tokenizers and Unigram intrinsic comparator, reproduce `phi`/`mu_e`/`mu_c` on a
held-out Turkish gold set, train matched BPE/MorphBPE models with complete
exposure and seed records, reproduce loss/BPB and paired downstream statistics,
and verify every published file against a manifest without relying on private
UHeM paths.

## Current Sources of Truth

- Method and raw-text inference contract:
  [`docs/tokenizer_tests/morphbpe_framework.md`](docs/tokenizer_tests/morphbpe_framework.md)
- Tokenizer metrics and sample sizes:
  [`docs/tokenizer_tests/tokenizer_metrics/`](docs/tokenizer_tests/tokenizer_metrics/)
- Model BPB inventory:
  [`docs/model_bpb_inventory.md`](docs/model_bpb_inventory.md)
- CETVEL model comparison:
  [`docs/cetvel_model_comparison.md`](docs/cetvel_model_comparison.md)
- Compact CETVEL artifact:
  [`artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/`](artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/)
- Tokenizer bundles:
  [`artifacts/tokenizers/`](artifacts/tokenizers/)
- Operational launcher index: [`runs/README.md`](runs/README.md)
