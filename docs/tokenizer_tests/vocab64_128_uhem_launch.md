# 64k/128k Tokenizer Ablation Launch

Date: 2026-06-12

This note records the larger-vocabulary tokenizer and model-training launch for
the Turkish MorphBPE ablation.

## Grid

| Vocab | Depth | Tokenizer family | Segmenter | Tokenizer name | Model tag |
| ---: | ---: | --- | --- | --- | --- |
| 65,536 | d16 | raw BPE | none | `bpe_65536` | `tr_d16_bpe_65536_chinchilla20` |
| 65,536 | d16 | MorphBPE | TRmorph | `morphbpe_trmorph_65536` | `tr_d16_morphbpe_trmorph_65536_chinchilla20` |
| 65,536 | d16 | MorphBPE | Zemberek | `morphbpe_zemberek_65536` | `tr_d16_morphbpe_zemberek_65536_chinchilla20` |
| 65,536 | d16 | MorphBPE | TurkishDelightNLP | `morphbpe_tdelight_65536` | `tr_d16_morphbpe_tdelight_65536_chinchilla20` |
| 131,072 | d12 | raw BPE | none | `bpe_131072` | `tr_d12_bpe_131072_chinchilla20` |
| 131,072 | d12 | MorphBPE | TRmorph | `morphbpe_trmorph_131072` | `tr_d12_morphbpe_trmorph_131072_chinchilla20` |
| 131,072 | d12 | MorphBPE | Zemberek | `morphbpe_zemberek_131072` | `tr_d12_morphbpe_zemberek_131072_chinchilla20` |
| 131,072 | d12 | MorphBPE | TurkishDelightNLP | `morphbpe_tdelight_131072` | `tr_d12_morphbpe_tdelight_131072_chinchilla20` |

## Launchers

- Tokenizer finalizer:
  [`runs/uhem_nakane_finalize_tokenizer_ablation.sbatch`](../../runs/uhem_nakane_finalize_tokenizer_ablation.sbatch)
- A100x4 model trainer:
  [`runs/uhem_nakane_a100x4_tokenizer_ablation.sbatch`](../../runs/uhem_nakane_a100x4_tokenizer_ablation.sbatch)
- Grid submitter:
  [`runs/uhem_submit_64k_128k_tokenizer_ablation.sh`](../../runs/uhem_submit_64k_128k_tokenizer_ablation.sh)

The submitter creates one CPU tokenizer job per grid row and one dependent
A100x4 model-training job per tokenizer job. GPU training keeps
`TOTAL_BATCH_SIZE=1048576` and `NUM_ITERATIONS=17100`, matching the 32k d20
token-position budget. This gives the same `17.93B` training token positions
while using d16 for 64k and d12 for 128k to keep total parameters near the
current 1B budget.

Tokenizer finalizer jobs write compact GitHub-ready bundles under:

```text
artifacts/tokenizers/<tokenizer_name>/
```

The large raw and segmented corpora stay on UHeM. The final bundles should be
copied back and committed after the corresponding tokenizer jobs complete.

## Submitted Jobs

Pending fill-in after Slurm submission.
