# SFT configuration screens: d32 with raw BPE 32k

These recovered posttraining results hold the tokenizer at **raw BPE, 32,768 tokens** and vary the training curriculum, learning rates and checkpoint. They supplement the [base-model tokenizer ablation](model_results.md); they are not additional tokenizer-family comparisons. All are descendants of the d32 FineWeb-2 model. The 41-case probe and 512-case production behavior suite are custom selection diagnostics, not CETVEL. Because these checks inform candidate selection, their pass rates are not independent held-out estimates of general capability.

Read-only UHeM recovery: `2026-11-09T12:47:17.506694+00:00`. [Source manifest](../../artifacts/uhem_sft_screens_2026-11-09/manifest.json), [flat CSV](sft_configuration_screens.csv), [full derived JSON](sft_configuration_screens.json). Original receipt bytes, source paths, SHA-256 hashes, code identities and checkpoint hashes are preserved. These summaries establish recorded results; this import does not re-execute their evaluators or verify current weight-file availability.

The common retained SFT source is `tr_d32_bpe32k_smoltalk2_turk_full_v9_then_fast_stage_b_v1`, checkpoint **84**. Its custom probe score is **9/41**. Low validation BPB on these small SFT corpora does not establish better general language modeling or downstream capability. Compare BPB only within a dataset and scoring protocol; it is not comparable to pretraining BPB or across the different curricula below.

## Dataset and protocol inventory

| Study | Training rows | Validation rows | Evaluation | Status |
| --- | ---: | ---: | --- | --- |
| Targeted repair v3 | 7,860 | 1,557 | 41-case selection probe | Nine verified checkpoints; retain source |
| Capability calibration v1 | 7,600 | 400 | Checkpoint validation BPB | Five verified checkpoints; incomplete screen |
| Final polish v1 | 21,414 | 1,557 | 41-case probe + 512-case production suite | Checkpoint complete; rejected by selection gates |
| Combined SmolTalk2-Turk SFT | 55,873 | 1,557 | Completion receipt only | Checkpoint 139 complete; no score inferred |

**Reading note.** These curricula have different training sizes and, for calibration, a different validation set. The inventory identifies what each screen measured; larger datasets and later checkpoint numbers are not evidence of better behavior. Repair/calibration/polish start from the retained SFT source, while the combined SmolTalk2 checkpoint is a separate run from the base model.

Targeted repair and final polish share the exact 1,557-row validation file (SHA-256 `2cae224e5cbfa297c8b6e8ebd3226bd99ed426e876a936053bdc1f670b8cfb6d`). Capability calibration uses a different 400-row set. Both curriculum manifests specify a 2,048-token maximum. Bounded-recovery data-row counts are not present in the selected receipts and are left unspecified.

## Bounded recovery: nine learning-rate/checkpoint candidates

The recovered selection reuses completed training and probe outputs after the initial selection step failed. The repaired decision is **retain source**: none of the nine candidates passes its selection gates. The arm names below are the recorded low/mid/high labels; exact checkpoint identities are retained in JSON.

| Arm | Checkpoint | SFT validation BPB ↓ | Probe passes / 41 ↑ | Qualified |
| --- | ---: | ---: | ---: | --- |
| low | 8 | 0.454580 | 8/41 | No |
| low | 16 | 0.450470 | 8/41 | No |
| low | 24 | 0.449236 | 6/41 | No |
| mid | 8 | 0.450021 | 6/41 | No |
| mid | 16 | 0.445096 | 6/41 | No |
| mid | 24 | 0.444001 | 6/41 | No |
| high | 8 | 0.444666 | 5/41 | No |
| high | 16 | 0.441746 | 6/41 | No |
| high | 24 | 0.441730 | 6/41 | No |

**Interpretation.** The high-rate arm at step 24 gives the lowest BPB (0.441730), but its probe score is only 6/41. The low-rate steps 8 and 16 lead the candidates at 8/41, still below the retained source at 9/41, and none passes the full selection gates. Here a lower validation loss does not identify the behaviorally better checkpoint.

Evidence: [probe scores](../../artifacts/uhem_sft_screens_2026-11-09/sources/bounded-recovery-v1r2__probe-selection-summary.json), [checkpoint BPB and hashes](../../artifacts/uhem_sft_screens_2026-11-09/sources/bounded-recovery-v1r2__recovery-admission-summary.json), [final decision](../../artifacts/uhem_sft_screens_2026-11-09/sources/bounded-recovery-v1r2__recovery-decision.json).

## Targeted repair: learning rates and checkpoint horizon

The table reports the declared matrix learning rate and the receipt’s effective initial smear learning rate. Other optimizer-group rates and the initial multiplier are preserved in the source receipts. All nine rows are diagnostic selection candidates; the recorded decision is **retain source**, with no probe-qualified candidate.

| Arm | Step | Matrix LR | Initial smear LR | SFT validation BPB ↓ | Probe passes / 41 ↑ | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| eff005 | 4 | 0.0010 | 0.0020 | 0.460415 | 8/41 | No |
| eff005 | 8 | 0.0010 | 0.0020 | 0.459071 | 8/41 | No |
| eff005 | 12 | 0.0010 | 0.0020 | 0.458667 | 8/41 | No |
| eff010 | 4 | 0.0020 | 0.0040 | 0.458865 | 8/41 | No |
| eff010 | 8 | 0.0020 | 0.0040 | 0.457161 | 8/41 | No |
| eff010 | 12 | 0.0020 | 0.0040 | 0.456851 | 8/41 | No |
| eff020 | 4 | 0.0040 | 0.0080 | 0.457116 | 8/41 | No |
| eff020 | 8 | 0.0040 | 0.0080 | 0.456307 | 9/41 | No |
| eff020 | 12 | 0.0040 | 0.0080 | 0.456318 | 10/41 | No |

**Interpretation.** The strongest-rate arm gives the lowest BPB at step 8 (0.456307) and the highest probe count at step 12 (10/41). Although 10/41 exceeds the source’s 9/41 overall count, the candidate fails the required category/selection gates, so the decision remains retain source. Extra steps and a higher rate therefore do not establish an acceptable improvement on this screen.

Evidence: [probe selection](../../artifacts/uhem_sft_screens_2026-11-09/sources/targeted-repair-screen-v3-probe-recovery-v1__probe-selection.json), [decision](../../artifacts/uhem_sft_screens_2026-11-09/sources/targeted-repair-screen-v3-probe-recovery-v1__screen-decision-summary.json). Each checkpoint verifier is linked through the source manifest.

## Capability calibration: completed checkpoints

This is a separate 7,600-training/400-validation-row curriculum. These rows establish checkpoint verification and its recorded validation BPB. They do not establish passing behavior probes or a selected release. The `eff040-16` summary is empty; no value is inferred for it or other unreported arms.

| Arm | Step | Matrix LR | Initial smear LR | SFT validation BPB ↓ | Verification |
| --- | ---: | ---: | ---: | ---: | --- |
| eff020 | 8 | 0.0040 | 0.0080 | 0.575542 | Pass |
| eff020 | 16 | 0.0040 | 0.0080 | 0.379709 | Pass |
| eff020 | 24 | 0.0040 | 0.0080 | 0.189516 | Pass |
| eff020 | 32 | 0.0040 | 0.0080 | 0.105042 | Pass |
| eff040 | 8 | 0.0080 | 0.0160 | 0.398625 | Pass |

**Interpretation.** Within eff020, additional updates reduce validation BPB from 0.575542 at step 8 to 0.105042 at step 32. At step 8, eff040 has lower BPB than eff020. These are loss-only observations: the imported receipts do not show a passing behavior probe, and the incomplete eff040 trajectory prevents a full rate-by-horizon comparison. The lowest BPB row is not a selected best model.

## Final polish: source versus candidate

The candidate is `tr_d32_bpe32k_final_instruct_polish_v1`, checkpoint **64**. Both checkpoints use raw BPE 32k. The saved selection rejects the candidate and retains source checkpoint 84. Pass counts below are custom behavior checks. Diagnostics such as echo and repetition are independent counts and need not sum to the suite size.

| Model | Suite | Passes ↑ | Pass rate ↑ | Empty ↓ | Echo ↓ | Repetition ↓ | Assistant-end termination |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Source | probe | 9/41 | 21.95% | 2 | 2 | 2 | 40 |
| Source | production | 68/512 | 13.28% | 0 | 179 | 15 | 490 |
| Candidate | probe | 5/41 | 12.20% | 0 | 3 | 4 | 37 |
| Candidate | production | 36/512 | 7.03% | 0 | 163 | 34 | 474 |

**Interpretation.** The retained source wins both behavior suites: 9 versus 5 passes on the probe and 68 versus 36 on the production suite. The candidate reduces prompt-echo counts in the production suite but increases repetition and loses overall passes, so its selection rejection follows the measured behavior. This is a post-training configuration result on one tokenizer, not a tokenizer-family comparison.

Evidence: [selection receipt](../../artifacts/uhem_sft_screens_2026-11-09/sources/final-polish-v1__selection-summary.json) and [checkpoint/data receipt](../../artifacts/uhem_sft_screens_2026-11-09/sources/final-polish-v1__checkpoint-summary.json).

## Completed checkpoint without an imported benchmark

`tr_d32_bpe32k_final_combined_smoltalk2_turk_v1` checkpoint **139** has a completed receipt, one-rank optimizer inventory, and training/validation manifest hashes. This establishes the saved SFT checkpoint, not a measured improvement. No benchmark value is inferred from completion. Evidence: [completion receipt](../../artifacts/uhem_sft_screens_2026-11-09/sources/final-combined-smoltalk2-turk-sft-v1__checkpoint-summary.json).

## Coverage and limitations

- These are selected recoverable configuration screens from UHeM, not a claim that every operational pilot or intermediate checkpoint has been exhaustively audited.
- Repeated checks of the same checkpoint and selection recovery do not count as independent training runs or seed repetitions.
- The initial bounded-recovery selection summary and capability `eff040-16` summary were empty at recovery; both are recorded as unavailable in the manifest. Their missing values are not zero.
- No matched multi-tokenizer SFT comparison was recovered. These d32/raw-BPE32k screens stay separate from the eleven-model tokenizer comparison and the separate CETVEL protocol results.
- Rebuild with `python -m scripts.build_sft_configuration_results`. The builder verifies all original-file SHA-256 values, checkpoint steps, probe denominators and verification statuses before emitting the tables.
