# CETVEL Model Comparison

This page tracks model-facing CETVEL evidence for the tokenizer ablation. It is separate from tokenizer-only diagnostics: these rows compare trained base models at the same checkpoint step, within each vocabulary/depth tier.

## Current Comparable Slice

The completed comparable slice is CETVEL core tasks 01-12 for 32k/d20, 64k/d16, and 128k/d12 base-model rows. The 32k TurkishDelightNLP tokenizer has no full d20 checkpoint and is therefore not included as a model row.

All completed rows use model step `17100`. Final validation BPB comes from final checkpoint metadata, lowest validation BPB comes from `loop_state.min_val_bpb`, and final train loss comes from the last printed training step (`step 17099/17100`).

The macro score averages the 11 classification/loglikelihood tasks. `xquad_tr` is reported separately because it is F1 rather than accuracy. CETVEL speed uses the progress-log `total_elapsed` at the end of the common core-12 slice divided by `39,441` expanded effective examples. Deltas are computed against raw BPE within the same vocabulary/depth tier.

| Vocab | Run | Tokenizer | Segmenter | CETVEL job | Elapsed | ex/s up | Speed vs raw | Val BPB | Lowest BPB | Train loss | Core-11 macro | Delta | XQuAD F1 | Delta | Source |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32k | raw BPE d20 | `bpe_32k` | none | `493293` | 50m20s | 13.06 | 1.000x | 0.6232 | 0.6232 | 2.4899 | 0.4514 | +0.0000 | 3.0985 | +0.0000 | [raw subset](../artifacts/cetvel_base_subset_2026-06-09_job493293/) |
| 32k | MorphBPE + TRmorph d20 | `morphbpe_trmorph_32k` | TRmorph | `494056` | 52m38s | 12.49 | 0.956x | 0.6266 | 0.6266 | 2.0106 | 0.4541 | +0.0027 | 3.4786 | +0.3801 | [ablation artifact](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) |
| 32k | MorphBPE + Zemberek d20 | `morphbpe_zemberek_32k` | Zemberek | `494057` | 50m30s | 13.02 | 0.997x | 0.6250 | 0.6250 | 2.3227 | 0.4618 | +0.0104 | 3.2633 | +0.1648 | [ablation artifact](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) |
| 64k | raw BPE d16 | `bpe_64k` | none | `496898` | 43m27s | 15.13 | 1.000x | 0.6409 | 0.6409 | 2.5812 | 0.4590 | +0.0000 | 2.8576 | +0.0000 | [ablation artifact](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) |
| 64k | MorphBPE + TRmorph d16 | `morphbpe_trmorph_64k` | TRmorph | `496899` | 43m09s | 15.23 | 1.007x | 0.6521 | 0.6521 | 2.2754 | 0.4532 | -0.0058 | 3.2778 | +0.4202 | [ablation artifact](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) |
| 64k | MorphBPE + Zemberek d16 | `morphbpe_zemberek_64k` | Zemberek | `496900` | 42m45s | 15.38 | 1.016x | 0.6514 | 0.6514 | 2.3013 | 0.4568 | -0.0022 | 2.8956 | +0.0380 | [ablation artifact](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) |
| 64k | MorphBPE + TurkishDelightNLP d16 | `morphbpe_tdelight_64k` | TurkishDelightNLP | `496901` | 40m52s | 16.09 | 1.063x | 0.6510 | 0.6510 | 2.4869 | 0.4567 | -0.0023 | 3.3280 | +0.4704 | [ablation artifact](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) |
| 128k | raw BPE d12 | `bpe_128k` | none | `496902` | 35m13s | 18.67 | 1.000x | 0.6749 | 0.6749 | 3.0976 | 0.4651 | +0.0000 | 2.2674 | +0.0000 | [ablation artifact](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) |
| 128k | MorphBPE + TRmorph d12 | `morphbpe_trmorph_128k` | TRmorph | `496903` | 35m09s | 18.70 | 1.002x | 0.6917 | 0.6917 | 2.5947 | 0.4503 | -0.0148 | 2.3517 | +0.0843 | [ablation artifact](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) |
| 128k | MorphBPE + Zemberek d12 | `morphbpe_zemberek_128k` | Zemberek | `496904` | 35m19s | 18.61 | 0.997x | 0.6940 | 0.6940 | 2.6000 | 0.4618 | -0.0033 | 2.9685 | +0.7011 | [ablation artifact](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) |
| 128k | MorphBPE + TurkishDelightNLP d12 | `morphbpe_tdelight_128k` | TurkishDelightNLP | `496905` | 33m25s | 19.67 | 1.054x | 0.6820 | 0.6820 | 2.6477 | 0.4481 | -0.0170 | 2.2498 | -0.0176 | [ablation artifact](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/) |

## Tier Takeaways

- `32k`: best core-11 macro is `morphbpe_zemberek_32k` (0.4618); best XQuAD F1 is `morphbpe_trmorph_32k` (3.4786); best validation BPB is `bpe_32k` (0.6232).
- `64k`: best core-11 macro is `bpe_64k` (0.4590); best XQuAD F1 is `morphbpe_tdelight_64k` (3.3280); best validation BPB is `bpe_64k` (0.6409).
- `128k`: best core-11 macro is `bpe_128k` (0.4651); best XQuAD F1 is `morphbpe_zemberek_128k` (2.9685); best validation BPB is `bpe_128k` (0.6749).

## Per-Task Core Comparison

Task-level values for all rows are stored in [`../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/metrics_summary.json`](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/metrics_summary.json) under `task_metrics`. The earlier 32k-only task table remains archived in [`../artifacts/cetvel_core12_model_comparison_2026-06-12`](../artifacts/cetvel_core12_model_comparison_2026-06-12/).

## Reading The Result

- The result is mixed rather than a uniform MorphBPE win.
- Raw BPE has the best validation BPB in all completed vocabulary tiers.
- Zemberek MorphBPE is best on the 32k core-11 macro, while raw BPE is best on the 64k and 128k core-11 macro.
- MorphBPE variants still improve selected task slices, especially XQuAD in some tiers: TRmorph at 32k, TurkishDelightNLP at 64k, and Zemberek at 128k.
- End-to-end CETVEL throughput generally improves at larger vocabularies because the d12/d16 models are shallower than d20; compare speed mainly within the same vocabulary/depth tier.
- These are pre-SFT base-model results. They should guide candidate selection, not final assistant-quality claims.
