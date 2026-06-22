# LLM Model BPB Inventory

This page records the current Turkish nanochat base-model checkpoints and their
byte-normalized validation losses. BPB values come from each completed run's
`meta_017100.json` checkpoint metadata on UHeM. For the completed rows below,
the final validation BPB equals `loop_state.min_val_bpb`, meaning validation BPB
was still improving at the final saved checkpoint.

## Current Checkpoints

| Vocab | Depth | Tokenizer | Model tag | Step | Final val BPB | Lowest val BPB | Current status |
| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 32,768 | d20 | `bpe_32768` | `tr_d20_bpe_32768_chinchilla20` | 17100 | 0.6232 | 0.6232 | CETVEL core compared; tasks 01-13 archived |
| 32,768 | d20 | `morphbpe_trmorph_32768` | `tr_d20_morphbpe_trmorph_32768_chinchilla20` | 17100 | 0.6266 | 0.6266 | CETVEL core compared |
| 32,768 | d20 | `morphbpe_zemberek_32768` | `tr_d20_morphbpe_zemberek_32768_chinchilla20` | 17100 | 0.6250 | 0.6250 | CETVEL core compared |
| 32,768 | d20 | `morphbpe_tdelight_32768` | `tr_d20_morphbpe_tdelight_32768_chinchilla20` | - | - | - | tokenizer exists; no full LLM checkpoint found |
| 65,536 | d16 | `bpe_65536` | `tr_d16_bpe_65536_chinchilla20` | 17100 | 0.6409 | 0.6409 | trained; CETVEL pending |
| 65,536 | d16 | `morphbpe_trmorph_65536` | `tr_d16_morphbpe_trmorph_65536_chinchilla20` | 17100 | 0.6521 | 0.6521 | trained; CETVEL pending |
| 65,536 | d16 | `morphbpe_zemberek_65536` | `tr_d16_morphbpe_zemberek_65536_chinchilla20` | 17100 | 0.6514 | 0.6514 | trained; CETVEL pending |
| 65,536 | d16 | `morphbpe_tdelight_65536` | `tr_d16_morphbpe_tdelight_65536_chinchilla20` | 17100 | 0.6510 | 0.6510 | trained; CETVEL pending |
| 131,072 | d12 | `bpe_131072` | `tr_d12_bpe_131072_chinchilla20` | 17100 | 0.6749 | 0.6749 | trained; CETVEL pending |
| 131,072 | d12 | `morphbpe_trmorph_131072` | `tr_d12_morphbpe_trmorph_131072_chinchilla20` | 17100 | 0.6917 | 0.6917 | trained; CETVEL pending |
| 131,072 | d12 | `morphbpe_zemberek_131072` | `tr_d12_morphbpe_zemberek_131072_chinchilla20` | 17100 | 0.6940 | 0.6940 | trained; CETVEL pending |
| 131,072 | d12 | `morphbpe_tdelight_131072` | `tr_d12_morphbpe_tdelight_131072_chinchilla20` | 17100 | 0.6820 | 0.6820 | trained; CETVEL pending |

## Reading The BPB Matrix

Validation BPB is the comparable loss metric across tokenizers because it is
normalized by bytes rather than tokenizer-specific prediction units. The raw
BPE 32k d20 model currently has the best checked-in final validation BPB
overall. The larger-vocabulary d16/d12 rows are trained and useful for the
fixed-parameter-budget ablation, but they still need CETVEL comparison before
they can support a downstream quality claim.

The 32k TurkishDelightNLP row is intentionally listed as tokenizer-only: the
tokenizer and raw tokenizer metrics exist on UHeM, but no full d20 base-model
checkpoint was found under its expected `base_checkpoints` path.

## UHeM Source Metadata

```text
/ari/users/nunal/nanochat-turk-d20-bpe32k/base_checkpoints/tr_d20_bpe_32768_chinchilla20/meta_017100.json
/ari/users/nunal/nanochat-turk-morphbpe-trmorph-32768/base_checkpoints/tr_d20_morphbpe_trmorph_32768_chinchilla20/meta_017100.json
/ari/users/nunal/nanochat-turk-morphbpe-zemberek-32768/base_checkpoints/tr_d20_morphbpe_zemberek_32768_chinchilla20/meta_017100.json
/ari/users/nunal/nanochat-turk-bpe-65536/base_checkpoints/tr_d16_bpe_65536_chinchilla20/meta_017100.json
/ari/users/nunal/nanochat-turk-morphbpe-trmorph-65536/base_checkpoints/tr_d16_morphbpe_trmorph_65536_chinchilla20/meta_017100.json
/ari/users/nunal/nanochat-turk-morphbpe-zemberek-65536/base_checkpoints/tr_d16_morphbpe_zemberek_65536_chinchilla20/meta_017100.json
/ari/users/nunal/nanochat-turk-morphbpe-tdelight-65536/base_checkpoints/tr_d16_morphbpe_tdelight_65536_chinchilla20/meta_017100.json
/ari/users/nunal/nanochat-turk-bpe-131072/base_checkpoints/tr_d12_bpe_131072_chinchilla20/meta_017100.json
/ari/users/nunal/nanochat-turk-morphbpe-trmorph-131072/base_checkpoints/tr_d12_morphbpe_trmorph_131072_chinchilla20/meta_017100.json
/ari/users/nunal/nanochat-turk-morphbpe-zemberek-131072/base_checkpoints/tr_d12_morphbpe_zemberek_131072_chinchilla20/meta_017100.json
/ari/users/nunal/nanochat-turk-morphbpe-tdelight-131072/base_checkpoints/tr_d12_morphbpe_tdelight_131072_chinchilla20/meta_017100.json
```
