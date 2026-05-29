# Turkish Foundation Pipeline

This branch trains nanochat base models on FineWeb-2 Turkish (`tur_Latn`) and
evaluates them with CETVEL. It intentionally stops at foundation/base models;
SFT will be added after the base pipeline is stable.

## Data

The default pretraining source is:

`hf://datasets/HuggingFaceFW/fineweb-2/data/tur_Latn/train`

`nanochat.dataset` resolves the Hugging Face tree, preserves the parquet order in
a local manifest, downloads the requested prefix, and always includes the
final remote shard as validation. This keeps the existing nanochat convention:
all local shards except the last are train, and the last shard is val.

Useful commands:

```bash
python -m nanochat.dataset --list-remote
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n -1
```

Set `NANOCHAT_DATASET_REVISION` or pass `--revision` to pin a Hugging Face commit.

## Training Horizon

Turkish foundation presets use Chinchilla-style `20` tokens per total parameter:

```bash
--target-param-data-ratio=20 --target-param-count=total
```

For comparison with upstream nanochat scaling-law experiments, `base_train.py`
also supports:

```bash
--target-param-count=scaling
```

Reports log all three ratios:

- `tokens / total params`
- `tokens / scaling params`
- `tokens / target params`

Plan candidate runs before launching:

```bash
python -m scripts.estimate_runs --vocab-size=32768
python -m scripts.estimate_runs --vocab-size=65536 --depths=12,16,20,24
```

Approximate Chinchilla-20 horizons for the default 32k raw BPE tokenizer:

| Depth | Total params | Target tokens |
| ---: | ---: | ---: |
| 4 | 36.7M | 0.73B |
| 8 | 125.8M | 2.52B |
| 12 | 286.3M | 5.73B |
| 16 | 536.9M | 10.74B |
| 20 | 896.5M | 17.93B |
| 24 | 1.384B | 27.68B |
| 26 | 1.682B | 33.64B |

These are planning horizons. Smoke runs intentionally override
`--num-iterations` and train on far fewer tokens.

Initial standard-tokenizer production runs:

| Vocab | Depth | Total params | Target tokens | Auto batch | Iterations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32k | 20 | 896.5M | 17.93B | 1,048,576 | 17,100 |
| 64k | 16 | 872.4M | 17.45B | 524,288 | 33,280 |

Nearest-over-1B alternatives are d21 for 32k (`1.099B @ 21.99B`) and d17 for
64k (`1.101B @ 22.02B`).

## Hardware

The model already supports A100-class CUDA training through bf16 and PyTorch
scaled dot-product attention. Hopper/H100-only paths such as FP8 and Flash
Attention 3 are optional and disabled by the A100 run profiles.

Use `HARDWARE_PROFILE=a100` for A100 jobs:

```bash
HARDWARE_PROFILE=a100 NPROC_PER_NODE=4 bash runs/turkish_foundation.sh
```

The A100 defaults are conservative:

- full-context attention: `WINDOW_PATTERN=L`
- no FP8: `USE_FP8=0`
- smaller per-device batches than H100 defaults

Use `HARDWARE_PROFILE=h100` only for H100/H200 runs where FP8 and the sliding
window profile are wanted.

## Tokenizers

The default implementation is upstream raw Rust BPE with the GPT-4-style split
pattern. Multiple tokenizer experiments can coexist by setting:

```bash
export NANOCHAT_TOKENIZER_NAME=bpe_32768
python -m scripts.tok_train --vocab-size=32768 --tokenizer-name=$NANOCHAT_TOKENIZER_NAME
```

Future tokenizer implementations should plug into the same naming convention:

- `bpe_32768`, `bpe_65536`, `bpe_131072`
- `morphbpe_*`
- `sentencepiece_*`
- morphology-aware segmentation variants

Use tokenizer metrics for ablations:

```bash
python -m scripts.tokenizer_metrics --max-docs=10000
```

This records bytes/token, chars/token, tokens/word fertility, and encode speed.

## CETVEL Suites

`scripts.cetvel_eval` defines three suites:

- `fast`: cheap iteration signal across MCQA, NLI, classification, QA, and PLU.
- `core`: major base-checkpoint suite focused on understanding-heavy tasks.
- `full`: `core` plus generation-heavy MT, summarization, and GEC diagnostics.

Examples:

```bash
python -m scripts.cetvel_eval --list-suites
python -m scripts.cetvel_eval --suite fast --limit 100 --model-tag tr_d12_bpe_32k
python -m scripts.cetvel_eval --suite core --model-tag tr_d24_bpe_32k_chinchilla20
```

Use `--auto-setup` to clone CETVEL and install its lm-evaluation-harness
submodule into the active environment.

## Run Presets

```bash
bash runs/turkish_smoke.sh
bash runs/turkish_foundation.sh
bash runs/turkish_ablation.sh
```

`turkish_foundation.sh` defaults to all FineWeb-2 Turkish shards and d24 with
Chinchilla-20 total-parameter training. Override with environment variables such
as `DEPTH`, `TRAIN_SHARDS`, `VOCAB_SIZE`, `NPROC_PER_NODE`, and `RUN_CETVEL`.

For a Google Colab A100 smoke test, keep the token budget tiny. Colab GPU type,
runtime length, and availability can change over time, so treat this as a
debug path rather than a production training path:

```bash
UV_EXTRA=gpu TRAIN_SHARDS=2 NUM_ITERATIONS=20 bash runs/turkish_smoke.sh
```

For a short A100 sanity run on the foundation script:

```bash
HARDWARE_PROFILE=a100 TRAIN_SHARDS=4 DEPTH=8 RUN_CETVEL=0 bash runs/turkish_foundation.sh
```

For UHeM-style Slurm clusters, start from:

```bash
sbatch runs/uhem_smoke_a100.sbatch
sbatch runs/uhem_a100.sbatch
```

Adjust the `#SBATCH` partition/account directives, module stack, and
`NANOCHAT_BASE_DIR` before submitting a production job. The default production
template targets one 4xA100 node; multi-node can be added later when allocation
pressure makes it worthwhile.

Recommended UHeM submissions:

```bash
# 13.0M params @ 0.26B tokens, standard 32k BPE path
sbatch runs/uhem_smoke_a100.sbatch

# 896.5M params @ 17.93B tokens
sbatch --export=ALL,VOCAB_SIZE=32768,NANOCHAT_TOKENIZER_NAME=bpe_32768,DEPTH=20,MODEL_TAG=tr_d20_bpe_32768_chinchilla20 runs/uhem_a100.sbatch

# 872.4M params @ 17.45B tokens
sbatch --export=ALL,VOCAB_SIZE=65536,NANOCHAT_TOKENIZER_NAME=bpe_65536,DEPTH=16,MODEL_TAG=tr_d16_bpe_65536_chinchilla20 runs/uhem_a100.sbatch
```

Cluster-specific details to fill in before the final runs can usually be
retrieved on the login node with:

```bash
sinfo -o "%P %a %l %D %c %m %G"
scontrol show partition
sacctmgr show assoc user=$USER format=Cluster,Account,User,Partition,QOS,MaxJobs,MaxSubmit,MaxWall,GrpTRES
module avail
module spider cuda
echo "$SCRATCH"
df -h "${SCRATCH:-$HOME}"
quota -s
```

The values we need to settle in the Slurm templates are:

- Slurm partition, account, and QoS names.
- GPU request syntax, for example `--gres=gpu:4` versus
  `--gres=gpu:a100:4` or `--gpus-per-node=4`.
- CUDA/Python module stack, or whether UHeM expects containerized jobs.
- Scratch filesystem path, quota, and purge policy for parquet shards,
  tokenizer files, checkpoints, and reports.
- Maximum walltime and maximum A100 count per job.
