# Türkçe Atlas SFT

This project uses `AlicanKiraz0/Turkce-Atlas-Instruct` as a format-only SFT
corpus for the `tr_d20_bpe_32768_chinchilla20` base checkpoint. The preparation
step never rewrites retained user or assistant text. It removes only three
assistant responses whose complete content is `S:`; the Atlas dataset card
identifies these examples as incomplete.

## Preparation policy

- Require every source row to contain exactly `system -> user -> assistant`.
- Remove the 654-character system preamble repeated in every Atlas row. Nanochat
  would otherwise fold it into every user prompt, adding about 45 million
  redundant tokens.
- Preserve the removed preamble in `source_system_prompt.txt` for auditability.
- Remove the three known-incomplete `S:` responses without altering any other
  dialogue text.
- Split by a seeded hash of exact user text, so duplicate prompts cannot appear
  in both training and validation.
- Emit the bare message-array JSONL expected by upstream nanochat's
  `CustomJSON`, with one complete `user -> assistant` conversation per row.

Run from the repository root:

```bash
python -m scripts.prepare_turkce_atlas_sft \
  --input '/Users/nurcunal/Downloads/AlicanKiraz0:Turkce-Atlas-Instruct/train.chat.jsonl' \
  --output-dir dev-ignore/turkce_atlas_sft \
  --validation-size 5000 \
  --split-seed 20260819 \
  --tokenizer-dir artifacts/tokenizers/bpe_32768 \
  --max-seq-len 2048
```

The generated `manifest.json` records source and output hashes, row counts, the
split policy, tokenizer hashes, length quantiles, and truncation statistics.
The output directory is intentionally ignored by Git because the JSONL files
are local training artifacts.

## Audited local artifact

The preparation above produced:

| Split | Rows | Formatted tokens | Assistant tokens | Max length |
|---|---:|---:|---:|---:|
| Train | 331,143 | 50,723,028 | 26,623,372 | 932 |
| Validation | 5,000 | 767,598 | 406,111 | 547 |
| Total | 336,143 | 51,490,626 | 27,029,483 | 932 |

No row exceeds the base model's 2,048-token context, so no assistant token is
lost to truncation. After excluding the three declared removals, the complete
source/output dialogue-pair multisets match exactly. Exact prepared hashes are:

- Train: `6cfbe686fd544cf506f13d8cc538fe0ef279c7d433e5c92a37e24fe5cfe0781d`
- Validation: `e803e9099abb121114ae615948459c06921568719b96a403da4d3b18afd49dad`
- Manifest: `be4eb5424b7d93cf0b44507134a867f0912a09e27d01c91977733982d0d64831`

The runtime preflight verifies these hashes, every JSONL row, split-group
isolation, tokenizer identity, the base model files/configuration, and at least
10 GB of free space before starting `torchrun`.

## SFT invocation

After placing the prepared directory on the training host, a conservative
one-epoch first run on the same four-A100 setup is:

```bash
export NANOCHAT_BASE_DIR="$HOME/nanochat-turk-d20-bpe32k"
export NANOCHAT_TOKENIZER_NAME=bpe_32768
export SFT_DATA_DIR="$NANOCHAT_BASE_DIR/sft_data/turkce_atlas_sft"

torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- \
  --model-tag=tr_d20_bpe_32768_chinchilla20 \
  --model-step=17100 \
  --load-optimizer=1 \
  --train-jsonl="$SFT_DATA_DIR/train.jsonl" \
  --val-jsonl="$SFT_DATA_DIR/validation.jsonl" \
  --data-manifest="$SFT_DATA_DIR/manifest.json" \
  --custom-json-lazy=1 \
  --max-seq-len=2048 \
  --device-batch-size=4 \
  --total-batch-size=1048576 \
  --embedding-lr=0.3 \
  --unembedding-lr=0.008 \
  --matrix-lr=0.02 \
  --eval-every=-1 \
  --eval-tokens=851968 \
  --chatcore-every=-1 \
  --run=tr-d20-bpe32k-atlas-sft-v1
```

`1,048,576` tokens is the batch size recorded by the base checkpoint; with four
GPUs, device batch 4, and context 2,048 this gives 32-way gradient accumulation.
The 331,143-row training split takes about 49 optimizer updates for one epoch.
`851,968` validation tokens is exactly 26 distributed validation batches and
covers the full validation split under the current packer. `--num-iterations`
is deliberately omitted (equivalent to `-1`), so nanochat stops after consuming
the training split once.

Use the guarded Slurm launcher in normal operation:

```bash
sbatch --export=ALL,MODE=smoke runs/uhem_nakane_a100x4_sft_atlas.sbatch
# Submit MODE=full only after the smoke job exits successfully.
sbatch --export=ALL,MODE=full runs/uhem_nakane_a100x4_sft_atlas.sbatch
```

Smoke mode performs exactly two optimizer updates and writes no checkpoint.
Full mode performs the one-epoch run and records the base checkpoint identity,
resolved configuration, prepared data/manifest hashes, and tokenizer file
hashes inside the final SFT metadata. It refuses to overwrite an existing SFT
model unless `ALLOW_SFT_OVERWRITE=1` is explicitly supplied. ChatCORE remains
disabled because its built-in suite is English; Turkish evaluation should be
run separately after the checkpoint is written.
