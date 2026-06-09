# nanochat-turk Project Report README

Last updated: 2026-06-09

This document is the running memory for the `nanochat-turk` project. It
summarizes the Codex project threads, especially `Plan Turkish tokenizer study`,
and turns the implementation work into report-ready notes. Use it as the first
source when writing the project report: it records what we decided, what we
implemented, why those choices were made, which results are already available,
and what still needs to be measured.

## Sources Summarized

This README condenses the following Codex threads and repository artifacts:

- `nanochat-turk`: original project planning, upstream refresh, Turkish
  foundation pipeline, A100/UHeM planning, branch cleanup.
- `Plan Turkish tokenizer study`: MorphBPE study design, segmenter evaluation,
  local judge pipeline, raw-text MorphBPE implementation, tokenizer guardrails,
  and UHeM MorphBPE launch preparation.
- `Find UHeM smoke test guide`: UHeM smoke and production runs, CETVEL
  evaluation debugging, W&B/progress logging, base-model benchmark decisions.
- `Teach terminal navigation commands`: UHeM/Altay operational commands and
  job-monitoring conventions.
- Repository docs under `docs/`, launch scripts under `runs/`, morphology
  helpers under `nanochat/morphology/`, and current branch history on
  `nanochat-turkish`.

## One-Page Project Thesis

We are adapting upstream `karpathy/nanochat` into a Turkish foundation-model
pipeline and a controlled Turkish tokenizer study.

The base project trains nanochat decoder-only base models on FineWeb-2 Turkish
(`tur_Latn`) using reproducible parquet ordering, Chinchilla-style `20` training
tokens per total model parameter, A100/UHeM-friendly training profiles, and
CETVEL as the Turkish benchmark suite.

The tokenizer research question is:

> Under an approximately fixed 900M-1B parameter Turkish LLM budget and fixed
> training-token budget, does morphology-aware tokenization improve tokenizer
> metrics, validation BPB, and CETVEL performance compared with raw BPE?

The current baseline is:

- raw Rust BPE, `bpe_32768`;
- model tag `tr_d20_bpe_32768_chinchilla20`;
- depth `20`;
- about `896.5M` total parameters;
- about `17.93B` training token positions;
- FineWeb-2 Turkish raw text;
- CETVEL evaluation before SFT.

The first MorphBPE target is:

- tokenizer name `morphbpe_trmorph_32768`;
- same raw FineWeb-2 Turkish text during LLM pretraining;
- segmentation used only to constrain BPE merge learning;
- no runtime segmenter required for user prompts or CETVEL prompts.

## Main Project Plan

1. Refresh the fork from current upstream nanochat.
   We rebuilt the project on a fresh `nanochat-turkish` branch from current
   `karpathy/master`, rather than merging the old stale Turkish branches. This
   preserved the modern nanochat training stack while selectively reintroducing
   Turkish-specific ideas. After a later cleanup request, old obsolete remote
   branches were removed and GitHub `master` was fast-forwarded to the Turkish
   work.

2. Build a Turkish base-model pipeline first.
   The project intentionally starts with foundation/base models, not SFT. The
   reason is scientific control: pretraining quality, tokenizer quality, BPB,
   and base CETVEL behavior should be measured before instruction tuning adds
   another data and formatting confound.

3. Use FineWeb-2 Turkish as the pretraining corpus.
   `nanochat.dataset` now targets
   `HuggingFaceFW/fineweb-2/data/tur_Latn/train`, preserves the Hugging Face
   parquet order in a local manifest, and follows nanochat's convention that the
   final shard is validation.

4. Use true Chinchilla-20 over total parameters for Turkish presets.
   Upstream nanochat's historical ratio logic uses a "scaling params" subset.
   We added `--target-param-count=total` and use `--target-param-data-ratio=20`
   for Turkish foundation runs, while logging total/scaling/target ratios so
   later tables are honest.

5. Keep raw BPE as the first baseline.
   We kept upstream Rust BPE as the baseline tokenizer and added
   `NANOCHAT_TOKENIZER_NAME` so multiple tokenizer artifacts can coexist safely.

6. Replace English-centric evaluation with CETVEL.
   We added a nanochat lm-evaluation-harness adapter and `scripts/cetvel_eval.py`
   suites: `fast`, `core`, and `full`. Base-model reporting should emphasize
   selection/loglikelihood tasks, with generation-heavy tasks treated as
   diagnostic until SFT.

7. Develop a Turkish morphology-aware tokenizer ablation.
   The study compares raw BPE with MorphBPE variants from TRmorph, Zemberek,
   and TurkishDelightNLP. Hybrid and pre-segmented controls remain useful, but
   the primary paper method is raw-text MorphBPE.

## What We Implemented

### Repository And Branch State

- Created and worked on `nanochat-turkish`.
- Pushed the Turkish branch to `nurcunal/nanochat-turk`.
- Fast-forwarded GitHub `master` to the same content as `nanochat-turkish`.
- Deleted old remote branches after the later branch-cleanup request:
  `turk`, `turkish`, and `turkish-synthetic`.

Important early commits:

- `e1a490c Adapt nanochat for Turkish foundation runs`
- `b5171c` corrected the production defaults to d20/d16 under-1B runs

### Data Layer

Implemented in `nanochat/dataset.py` and documented in
`docs/turkish_foundation.md`.

- Default dataset is FineWeb-2 Turkish `tur_Latn`.
- The Hugging Face tree order is preserved in `fineweb2_manifest.json`.
- `python -m nanochat.dataset -n -1` downloads all Turkish shards.
- The final remote shard is always included for validation.
- `NANOCHAT_DATA_DIR` and `NANOCHAT_TEXT_COLUMN` allow raw and segmented
  corpora to share the same dataloader path.

Why this matters for the report:

- Reproducibility depends on deterministic shard order.
- The tokenizer ablation must keep data source and train/val convention fixed.
- Raw-text and segmented-corpus experiments can be compared without duplicating
  loader logic.

### Training Horizon And A100 Profile

Implemented in `scripts/base_train.py`, `scripts/estimate_runs.py`, and run
scripts under `runs/`.

- Added total-parameter Chinchilla targeting.
- Added `scripts/estimate_runs.py` to compute params, target tokens, batch, and
  iteration counts.
- Added A100-friendly defaults:
  - bf16;
  - full-context attention: `WINDOW_PATTERN=L`;
  - no FP8;
  - conservative device batch sizes.
- Added UHeM/Altay Slurm scripts for smoke, prep, production, CETVEL, upload,
  and MorphBPE.

Primary run sizes:

| Vocab | Depth | Params | Token positions | Purpose |
| ---: | ---: | ---: | ---: | --- |
| 32k | d20 | 896.5M | 17.93B | first raw-BPE baseline |
| 64k | d16 | 872.4M | 17.45B or matched 17.93B | matched larger-vocab baseline |
| 128k | d12 | 890.2M | matched 17.93B | later larger-vocab ablation |

### UHeM Smoke Test

Completed smoke artifact:

`artifacts/uhem_smoke_2026-06-07_job492393/`

Summary:

- Altay/UHeM job `492393`.
- State `COMPLETED`, exit `0:0`.
- 1x A100 80GB, partition `gpu2dq`.
- Model tag `tr_d2_bpe_32768_uhem_smoke_chinchilla20`.
- Depth `2`, `12.98M` parameters.
- Training tokens `259,522,560`.
- Tokens/total-params ratio `19.9999`.
- Minimum validation BPB `1.239557`.
- Base eval val BPB `1.239951`.

Report use:

- This is not a model-quality result. It is infrastructure evidence that the
  Turkish data download, tokenizer training, base pretraining, checkpointing,
  evaluation, and artifact capture work on UHeM.

### Tokenizer Artifact Safety

Implemented in `nanochat/tokenizer.py`, `scripts/base_train.py`,
`nanochat/checkpoint_manager.py`, and `scripts/upload_base_checkpoint_to_hf.py`.

- Tokenizers are name-scoped under:
  `$NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME/`
- New checkpoints record:
  - `tokenizer_name`;
  - `tokenizer_dir`;
  - `tokenizer_config`.
- Checkpoint loading prefers the tokenizer recorded in metadata over the current
  shell environment.
- HF upload fails if an explicit tokenizer name conflicts with checkpoint
  metadata.

Why this matters:

- Same-vocab tokenizer ablations can silently fail if a 32k MorphBPE model is
  evaluated with a 32k raw BPE tokenizer. These guardrails make that mistake
  much harder.

Caveat:

- The earliest raw-BPE baseline may have been launched before this metadata
  patch. For that run, evaluation/upload should explicitly set
  `NANOCHAT_TOKENIZER_NAME=bpe_32768` or pass `--tokenizer-name bpe_32768`.

### CETVEL Integration

Implemented in `scripts/cetvel_eval.py` and `nanochat/lm_eval_nanochat.py`.

Added:

- lm-evaluation-harness model adapter for nanochat checkpoints.
- `fast`, `core`, and `full` CETVEL suites.
- CETVEL auto-setup and local task-config patching for current Hugging Face
  dataset aliases.
- `datasets==2.19.2` compatibility handling for CETVEL's legacy dataset
  scripts.
- Task renames for built-in lm-eval name collisions.
- Task-by-task progress mode with partial JSON results.
- Per-generation progress logging for generation-heavy tasks.
- W&B logging at benchmark start and after task completion.
- KV-cache generation through `Engine.generate`, replacing the old naive
  full-context recompute path.
- NumPy scalar/array JSON serialization fixes.

Key commits:

- `9fe5dd4 Add CETVEL task progress logging`
- `5dbb480 Log CETVEL generation progress`
- `0433adc Use KV cache for CETVEL generation`
- `d1bced3 Stream CETVEL task progress to wandb`
- `cc12670 Serialize CETVEL numpy metrics`

Base-model evaluation decision:

- Benchmark base models before SFT.
- Headline base evidence should use validation BPB plus CETVEL
  selection/loglikelihood tasks.
- CETVEL generation tasks are useful diagnostics for base models but should be
  headline results mainly after SFT.

Tasks 1-11 are the clean base-model subset:

`exams_tr`, `belebele_tr`, `turkish_plu`, `cetvel_xcopa_tr`,
`cetvel_xnli_tr`, `mnli_tr`, `snli_tr`, `news_cat`, `offenseval_tr`,
`trclaim19`, `xfact_tr`.

Tasks 12-20 are generation-heavy and more SFT-aligned:

`xquad_tr`, `tquad`, `mkqa_tr`, WMT, `mlsum_tr`, `xlsum_tr`,
`wiki_lingua_tr`, `gecturk_generation`.

Preliminary raw-BPE CETVEL evidence from the live UHeM run:

- Early tasks were verified as real, not skipped.
- Example metrics observed:
  - `exams_tr` accuracy about `0.2799`;
  - `belebele_tr` accuracy about `0.2522`;
  - `turkish_plu` accuracy about `0.4605`;
  - `cetvel_xcopa_tr` accuracy about `0.6180`;
  - `cetvel_xnli_tr` accuracy about `0.3327`, near 3-way chance.
- The active plan in the benchmark thread was to stop after `tquad` completes
  and upload the partial base-model results to HF with clear subset labeling.

Do not call partial results "full CETVEL" unless all 20 tasks finish.

### Morphology Segmenter Infrastructure

Implemented in:

- `nanochat/morphology/segmenters.py`
- `scripts/morph_benchmark.py`
- `scripts/morph_segment_corpus.py`
- `scripts/zemberek_segment_cmd.py`
- `scripts/tdelight_segment_cmd.py`
- `scripts/morph_judge_pack.py`
- `scripts/morph_judge_score.py`
- `scripts/morph_codex_local_judge.py`

Segmenter backends:

- `trmorph`: FST via `flookup` and `segment.fst`.
- `zemberek`: Python wrapper around Zemberek morphology.
- `tdelight`: TurkishDelightNLP via command wrapper or REST endpoint.
- `identity`: no-segmentation control.

The benchmark methodology:

- Count word-like forms in the first FineWeb-2 Turkish shard.
- Segment unique word types.
- Weight results by corpus frequency.
- Accept only segmentations whose pieces reconstruct the exact original
  surface form.
- Record throughput, split rate, pieces/word, fallback rate, and fallback
  reasons.

First-shard inventory:

| Field | Value |
| --- | ---: |
| Documents | 3,381,000 |
| Word-like tokens | 1,282,426,655 |
| Unique word forms | 10,829,544 |
| UTF-8 bytes | 10,981,374,221 |

Equal hash-100k screening results:

| Backend | Unique/s | Weighted split | Weighted fallback | Type fallback |
| --- | ---: | ---: | ---: | ---: |
| identity | ~510k | 0.000 | 0.000 | 0.000 |
| TRmorph | ~9.3k | 0.338 | 0.145 | 0.817 |
| Zemberek | ~3.3k | 0.316 | 0.425 | 0.233 |
| TurkishDelightNLP | ~1.5k | 0.362 | 0.000 | 0.000 |

Interpretation:

- TRmorph is the strongest judged segmenter when it returns usable analyses.
- TurkishDelightNLP has excellent coverage in the wrapper and zero fallback
  after surface realignment, but is slower.
- Zemberek is a conservative control and sometimes acceptable, but current
  judged quality does not make it the primary candidate.

### Blind Local Segmenter Judge

Implemented in `scripts/morph_codex_local_judge.py` and documented in
`docs/tokenizer_tests/codex_local_judge_results.md`.

Method:

- Generate a 500-item disagreement-focused judge pack from the same hash-100k
  sample.
- Hide backend names behind labels `A`, `B`, `C`, `D`.
- Judge only the word and candidate segmentations.
- Decode backend names only after all judgments are written.
- Score best-label and acceptable-label rates by backend.

Scores:

| Backend | Best rate | Acceptable rate |
| --- | ---: | ---: |
| TRmorph | 48.2% | 51.0% |
| TurkishDelightNLP | 34.6% | 43.8% |
| Zemberek | 17.2% | 40.0% |
| identity | 0.0% | 0.8% |

Limitations for report:

- This is not a native-speaker gold annotation.
- The pack is disagreement-focused, not corpus-representative.
- No sentence context was used.
- Ambiguous forms may require human annotation.

Report framing:

- Use this as a preliminary segmenter-quality screening step, not final proof.
- It supports carrying at least TRmorph and TurkishDelight into tokenizer
  ablations.

### MorphBPE Implementation

Implemented in:

- `nanochat/morphology/boundary.py`
- `nanochat/morphology/morphbpe.py`
- `scripts/tok_train.py`
- `docs/tokenizer_tests/morphbpe_framework.md`
- `tests/test_morphbpe_tokenizer.py`

Critical correction from the tokenizer thread:

- Pre-segmented BPE and MorphBPE are different.
- Pre-segmented BPE changes the model's text stream and may require runtime
  segmentation at inference.
- Paper-faithful MorphBPE uses segmentation only while training the merge table.
  The final tokenizer encodes raw text normally.

Our main method:

1. Segment words into surface morpheme spans.
2. Insert internal `U+E000` boundaries in a segmented training column.
3. During tokenizer training, split boundary-marked text into morpheme-internal
   chunks.
4. Let RustBPE train only on chunks, so it never counts cross-boundary adjacent
   pairs.
5. Save a standard raw-text tokenizer with:
   - `implementation = morphbpe`;
   - `training_boundary_semantics = merge_constraint_only`;
   - `requires_runtime_segmentation = false`;
   - `decode_strip = ""`.
6. Train and evaluate the LLM on raw FineWeb-2 Turkish text.

Control method:

- `preseg_bpe_*` trains and models boundary-marked text directly.
- It is useful for the paper because it tests whether gains come from the merge
  table or from explicitly showing morpheme boundaries to the LM.

Key commits:

- `cc51d9b Add MorphBPE boundary tokenizer framework`
- `3f77265 Implement raw-text MorphBPE training`

### UHeM MorphBPE Preflight

Implemented in:

- `runs/uhem_nakane_prepare_morphbpe_trmorph_32k.sbatch`
- `runs/uhem_nakane_a100x4_morphbpe_trmorph_32k.sbatch`

What was checked:

- Local tests passed: `29 passed, 10 skipped`.
- Local TRmorph smoke produced boundary-marked text.
- UHeM W&B verification succeeded as `nurcunal`.
- UHeM foma/flookup was built in a user-owned vendor directory.
- `segment.fst` was copied to UHeM.
- Remote TRmorph smoke worked:
  `evlerden geldik çalışıyorum` -> `ev ler den gel dik çalış ıyor um`.
- Remote tiny end-to-end MorphBPE smoke passed:
  - `implementation = morphbpe`;
  - `training_boundary_semantics = merge_constraint_only`;
  - `requires_runtime_segmentation = false`;
  - raw-text roundtrip OK.
- CPU prep memory was corrected from `256G` to `240G` because `cpu2dq` nodes
  advertise `250000M`.

Current next action from the tokenizer thread:

- Submit the CPU prep job first.
- It materializes the full TRmorph segmented corpus and trains the real
  `morphbpe_trmorph_32768` tokenizer.
- Do not start GPU training until the production tokenizer artifacts exist.
- Once prep completes, prefer a single 4xA100 node unless queue conditions
  improve; 2-node and 4-node starts were estimated much later when last checked.

Key commits:

- `2f35832 Add UHeM TRmorph MorphBPE run scripts`
- `dd456fb Fix UHeM MorphBPE prep memory request`

## Important Design Decisions And Why

### Base Before SFT

Decision:

- Evaluate base checkpoints first, then SFT later.

Reason:

- The primary tokenizer question is about pretraining. SFT introduces
  instruction data, prompt formatting, and generation behavior as confounds.
  Base BPB and selection-style CETVEL tasks are cleaner evidence for tokenizer
  and foundation-model quality.

Report wording:

> We report base-model CETVEL results as zero-shot diagnostics of pretraining
> quality. Generation-heavy CETVEL tasks are treated cautiously for base models
> and reserved for stronger interpretation after SFT.

### Fixed Token Positions, Not Fixed Raw Bytes

Decision:

- Primary full-scale runs fix training token positions around `17.93B`.

Reason:

- nanochat training is scheduled in tokens and comparable compute. But
  tokenizers differ in fertility, so fixed token positions do not imply equal
  raw bytes or equal documents seen.

Report caveat:

> We report raw bytes and document counts consumed for each tokenizer because a
> fixed token-position budget induces different raw-corpus exposure across
> tokenizers.

### Raw-Text MorphBPE As Main Method

Decision:

- Main MorphBPE tokenizers should encode raw Turkish text without runtime
  segmentation.

Reason:

- This matches the MorphBPE paper's practical claim: morphology constrains merge
  learning, while inference remains standard BPE. It also avoids requiring
  CETVEL/user prompts to be preprocessed by a segmenter.

Report caveat:

- A raw-text MorphBPE tokenizer cannot absolutely guarantee boundary-respecting
  inference for every ambiguous word, because it does not run a segmenter at
  inference. Its merge table is trained without cross-boundary evidence.

### Separate Segmenter Variants Before Hybrid

Decision:

- Treat TRmorph, TurkishDelightNLP, and Zemberek as separate first-class
  tokenizer variants before making a hybrid.

Reason:

- A hybrid requires many selection heuristics and reviewer-unfriendly degrees of
  freedom. Separate variants give cleaner causality: if TRmorph wins, we can
  say the TRmorph-constrained tokenizer won; if TurkishDelight wins, coverage
  may matter more than judged linguistic sharpness.

Hybrid status:

- Worth exploring later, likely TRmorph first with TurkishDelight fallback.
- Not the first paper-critical method.

## Current Report-Relevant Results

### Infrastructure Result

UHeM smoke job `492393` completed successfully. This validates the end-to-end
pipeline on Altay, but not model quality.

Use in report:

> Before full-scale training, we verified the pipeline with a depth-2 A100 smoke
> run, exercising FineWeb-2 Turkish download, tokenizer training, base
> pretraining, checkpointing, BPB evaluation, and artifact capture.

### Segmenter Readiness Result

The segmenter study has two layers:

1. Hash-100k quantitative screening.
2. 500-item blind local judge quality screening.

Best current segmenter for first MorphBPE run:

- TRmorph, because it wins the local blind judge most often and has good
  frequency-weighted coverage.

Strong second:

- TurkishDelightNLP, because it has zero fallback after surface realignment and
  strong acceptable-label rate.

Conservative control:

- Zemberek.

Use in report:

> We use segmenter benchmarks as a filtering step, not as final tokenizer
> evidence. The final claim must come from tokenizer metrics, validation BPB,
> and CETVEL scores after matched LLM training.

### Baseline Raw-BPE CETVEL Result

The raw-BPE d20 model benchmark was being monitored in the UHeM benchmark
thread. The optimized CETVEL run wrote task-level partial artifacts and reached
the generation-heavy section. The recommended reporting stance was:

- headline base-model benchmark: tasks 1-11;
- optionally include XQuAD/TQuAD as extractive QA diagnostics if completed;
- defer open generation, summarization, translation, and grammar correction
  headline claims to post-SFT models.

Use in report:

> For the base model, we emphasize selection/loglikelihood CETVEL tasks because
> they directly measure next-token likelihood over answer alternatives. We do
> not interpret poor open-generation performance as conclusive evidence against
> pretraining quality before SFT.

## Next Steps

Immediate:

1. Finish or stop the current raw-BPE CETVEL run according to the benchmark
   decision: stop after `tquad` if that is still the active plan, then upload
   the partial result set with explicit subset labeling.
2. Upload the raw-BPE d20 model/checkpoint/tokenizer/CETVEL artifacts to HF.
3. Submit the UHeM CPU prep job:
   `runs/uhem_nakane_prepare_morphbpe_trmorph_32k.sbatch`.
4. Verify the production artifacts:
   `$HOME/nanochat-turk-morphbpe-trmorph-32768/tokenizers/morphbpe_trmorph_32768`.
5. Launch the `morphbpe_trmorph_32768` d20 run on 1x 4A100 unless queue
   conditions improve.

Short-term tokenizer study:

1. Run tokenizer-only metrics for `bpe_32768` and `morphbpe_trmorph_32768`.
2. Materialize Zemberek and TurkishDelight segmented corpora.
3. Train `morphbpe_zemberek_32768` and `morphbpe_tdelight_32768`.
4. Run tokenizer-only metrics and boundary-violation metrics.
5. Decide whether all three should go to full d20 training or whether one is
   ruled out by tokenizer-only evidence.

Medium-term:

1. Train matched 64k and 128k tiers if compute allows.
2. Add SentencePiece BPE/unigram controls.
3. Create a human/native-speaker annotated segmentation subset.
4. Run SFT for the best base checkpoint(s).
5. Re-run full CETVEL after SFT and use generation tasks as final usability
   evidence.

## Report-Writing Prompts

Use these prompts when drafting the critical sections.

### Abstract Prompt

Write an abstract about a Turkish nanochat foundation-model pipeline and
tokenizer ablation study. Mention FineWeb-2 Turkish, Chinchilla-20 total-param
training, CETVEL evaluation, raw-BPE baseline, MorphBPE tokenizers constrained
by Turkish segmenters, and the goal of testing whether morphology-aware
tokenization improves base-model performance under matched compute.

### Introduction Prompt

Explain why Turkish is a useful case for tokenizer research: agglutinative
morphology creates long productive word forms, frequency-only BPE may learn
tokens that cross morpheme boundaries, and tokenizer fertility affects both
compute efficiency and downstream performance. Frame the project as a controlled
LLM ablation rather than only a tokenizer-intrinsic study.

### Dataset And Reproducibility Prompt

Describe the FineWeb-2 Turkish setup: `tur_Latn`, ordered parquet manifest,
final-shard validation convention, optional HF revision pinning, and the reason
we preserve exact shard order. Add that full raw/segmented corpora are stored
outside git while scripts, manifests, aggregate metrics, and docs are tracked.

### Model And Training Prompt

Describe the nanochat architecture as depth-controlled and compute-planned.
Explain our use of total-parameter Chinchilla-20 rather than upstream scaling
params. Include the d20/32k baseline: `896.5M` params and `17.93B` token
positions, A100 bf16 profile, no FP8, full attention, UHeM Slurm execution.

### Tokenizer Method Prompt

Contrast three tokenizer conditions:

- raw BPE baseline;
- MorphBPE main method, where segmentation constrains merge training but final
  inference uses raw text;
- pre-segmented BPE control, where boundary markers are part of the model text
  stream and stripped on decode.

Emphasize that main MorphBPE has `requires_runtime_segmentation = false`.

### Segmenter Evaluation Prompt

Explain the segmenter screening pipeline: first-shard word inventory,
frequency-weighted hash-100k runs, exact surface reconstruction requirement,
throughput/split/fallback metrics, and blind 500-item local judge pack. Report
TRmorph, TurkishDelight, and Zemberek as complementary candidates, not as a
final answer before LLM training.

### Evaluation Prompt

Describe CETVEL integration through lm-evaluation-harness. Explain the `fast`,
`core`, and `full` suites. For base models, justify focusing on
multiple-choice/loglikelihood tasks and treating generation-heavy tasks as
post-SFT or diagnostic.

### Results Prompt

Separate results by maturity:

1. Infrastructure smoke passed.
2. Segmenter screening results.
3. Raw-BPE baseline training/evaluation artifacts.
4. MorphBPE preflight readiness.
5. Full matched LLM ablation results once available.

Do not overclaim from partial CETVEL or local judge results.

### Limitations Prompt

List limitations:

- no native-speaker gold segmentation set yet;
- fixed token budget is not fixed raw-byte exposure;
- base-model generation tasks are hard to interpret before SFT;
- raw-text MorphBPE biases the merge table but does not enforce runtime
  morpheme boundaries;
- UHeM queue/hardware variability affects wall-clock comparisons.

### Future Work Prompt

Discuss SFT, full CETVEL after SFT, SentencePiece controls, hybrid segmenters,
human morphology annotation, matched-raw-byte robustness runs, and publishing
tokenizers/checkpoints/results on Hugging Face.

## File Map For Report Authors

- Main workflow: `docs/turkish_foundation.md`
- Tokenizer study plan: `docs/tokenizer_ablation_plan.md`
- MorphBPE implementation notes:
  `docs/tokenizer_tests/morphbpe_framework.md`
- Segmenter benchmark status:
  `docs/tokenizer_tests/segmenter_benchmark_status.md`
- Local judge results:
  `docs/tokenizer_tests/codex_local_judge_results.md`
- LLM/human judge workflow:
  `docs/tokenizer_tests/llm_judge_pipeline.md`
- TurkishDelight setup:
  `docs/tokenizer_tests/turkishdelight_setup.md`
- UHeM smoke artifact:
  `artifacts/uhem_smoke_2026-06-07_job492393/`
- Core training script:
  `scripts/base_train.py`
- CETVEL runner:
  `scripts/cetvel_eval.py`
- nanochat lm-eval adapter:
  `nanochat/lm_eval_nanochat.py`
- Tokenizer trainer:
  `scripts/tok_train.py`
- MorphBPE helpers:
  `nanochat/morphology/morphbpe.py`
- Segmenter wrappers:
  `nanochat/morphology/segmenters.py`
- UHeM raw-BPE production run:
  `runs/uhem_nakane_a100x4_d20_bpe32k.sbatch`
- UHeM TRmorph MorphBPE prep/run:
  `runs/uhem_nakane_prepare_morphbpe_trmorph_32k.sbatch`
  and `runs/uhem_nakane_a100x4_morphbpe_trmorph_32k.sbatch`

## Claim Boundaries

Safe current claims:

- The Turkish foundation pipeline exists and passed a UHeM smoke test.
- The repo supports FineWeb-2 Turkish ordered-shard pretraining.
- The repo supports CETVEL base-model evaluation with progress, W&B, partial
  artifacts, and KV-cache generation.
- The repo supports tokenizer-specific artifact naming and checkpoint tokenizer
  guardrails.
- Segmenter screening suggests TRmorph and TurkishDelight are the strongest
  candidates to carry forward.
- Raw-text MorphBPE training is implemented and smoke-tested.

Claims to avoid until more evidence exists:

- "MorphBPE improves Turkish LLM performance."
- "TRmorph is definitively the best tokenizer segmenter."
- "The base model has completed full CETVEL" unless all 20 tasks and artifacts
  are available.
- "Generation CETVEL scores measure final assistant quality" before SFT.
- "No Turkish morphology-aware tokenizer exists." Use the more defensible claim:
  controlled Turkish decoder-only LLM tokenizer ablation under matched budgets.
