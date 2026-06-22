# CETVEL Core-12 Tokenizer Ablation (2026-06-22)

Compact benchmark artifact for the Turkish tokenizer ablation. Raw CETVEL JSON files include large sample payloads and remain on UHeM; this directory stores reproducible summary metrics and source paths.

Deltas are computed against raw BPE within the same vocabulary/depth tier. Core-11 macro excludes `xquad_tr`; XQuAD F1 is reported separately.

## Summary

| Vocab | Depth | Run | Tokenizer | Job | Elapsed | ex/s | Speed vs raw | Val BPB | Lowest BPB | Train loss | Core-11 macro | Delta | XQuAD F1 | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32k | d20 | raw BPE | `bpe_32k` | `493293` | 50m20s | 13.06 | 1.000x | 0.6232 | 0.6232 | 2.4899 | 0.4514 | +0.0000 | 3.0985 | +0.0000 |
| 32k | d20 | MorphBPE + TRmorph | `morphbpe_trmorph_32k` | `494056` | 52m38s | 12.49 | 0.956x | 0.6266 | 0.6266 | 2.0106 | 0.4541 | +0.0027 | 3.4786 | +0.3801 |
| 32k | d20 | MorphBPE + Zemberek | `morphbpe_zemberek_32k` | `494057` | 50m30s | 13.02 | 0.997x | 0.6250 | 0.6250 | 2.3227 | 0.4618 | +0.0104 | 3.2633 | +0.1648 |
| 64k | d16 | raw BPE | `bpe_64k` | `496898` | 43m27s | 15.13 | 1.000x | 0.6409 | 0.6409 | 2.5812 | 0.4590 | +0.0000 | 2.8576 | +0.0000 |
| 64k | d16 | MorphBPE + TRmorph | `morphbpe_trmorph_64k` | `496899` | 43m09s | 15.23 | 1.007x | 0.6521 | 0.6521 | 2.2754 | 0.4532 | -0.0058 | 3.2778 | +0.4202 |
| 64k | d16 | MorphBPE + Zemberek | `morphbpe_zemberek_64k` | `496900` | 42m45s | 15.38 | 1.016x | 0.6514 | 0.6514 | 2.3013 | 0.4568 | -0.0022 | 2.8956 | +0.0380 |
| 64k | d16 | MorphBPE + TurkishDelightNLP | `morphbpe_tdelight_64k` | `496901` | 40m52s | 16.09 | 1.063x | 0.6510 | 0.6510 | 2.4869 | 0.4567 | -0.0023 | 3.3280 | +0.4704 |
| 128k | d12 | raw BPE | `bpe_128k` | `496902` | 35m13s | 18.67 | 1.000x | 0.6749 | 0.6749 | 3.0976 | 0.4651 | +0.0000 | 2.2674 | +0.0000 |
| 128k | d12 | MorphBPE + TRmorph | `morphbpe_trmorph_128k` | `496903` | 35m09s | 18.70 | 1.002x | 0.6917 | 0.6917 | 2.5947 | 0.4503 | -0.0148 | 2.3517 | +0.0843 |
| 128k | d12 | MorphBPE + Zemberek | `morphbpe_zemberek_128k` | `496904` | 35m19s | 18.61 | 0.997x | 0.6940 | 0.6940 | 2.6000 | 0.4618 | -0.0033 | 2.9685 | +0.7011 |
| 128k | d12 | MorphBPE + TurkishDelightNLP | `morphbpe_tdelight_128k` | `496905` | 33m25s | 19.67 | 1.054x | 0.6820 | 0.6820 | 2.6477 | 0.4481 | -0.0170 | 2.2498 | -0.0176 |

- `32k`: best core-11 macro is `morphbpe_zemberek_32k` (0.4618); best XQuAD F1 is `morphbpe_trmorph_32k` (3.4786); best validation BPB is `bpe_32k` (0.6232).
- `64k`: best core-11 macro is `bpe_64k` (0.4590); best XQuAD F1 is `morphbpe_tdelight_64k` (3.3280); best validation BPB is `bpe_64k` (0.6409).
- `128k`: best core-11 macro is `bpe_128k` (0.4651); best XQuAD F1 is `morphbpe_zemberek_128k` (2.9685); best validation BPB is `bpe_128k` (0.6749).

## Per-Task Metrics

Task-level values are stored in `metrics_summary.json` under `task_metrics`.

## Source

- Old 32k rows are carried forward from `artifacts/cetvel_core12_model_comparison_2026-06-12`.
- New 64k/128k rows come from UHeM CETVEL jobs `496898` through `496905`.
- `morphbpe_tdelight_128k` had a complete `cetvel_core_results.json` at import time while Slurm was still finishing wandb/report upload; the source path and job id are recorded in `metrics_summary.json`.
