#!/bin/bash

# Slurm template for UHeM-style A100 foundation jobs. Adjust
# partition/account directives to match your allocation before submitting with:
#   sbatch runs/uhem_a100.sbatch
#
# Approximate 1B standard-tokenizer runs:
#   sbatch --export=ALL,VOCAB_SIZE=32768,NANOCHAT_TOKENIZER_NAME=bpe_32768,DEPTH=21,MODEL_TAG=tr_d21_bpe_32768_chinchilla20 runs/uhem_a100.sbatch
#   sbatch --export=ALL,VOCAB_SIZE=65536,NANOCHAT_TOKENIZER_NAME=bpe_65536,DEPTH=17,MODEL_TAG=tr_d17_bpe_65536_chinchilla20 runs/uhem_a100.sbatch

#SBATCH --job-name=nanochat-tr-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

mkdir -p logs

# Uncomment/adapt these for the UHeM software stack if needed.
# module purge
# module load cuda/12.8

export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${SCRATCH:-$HOME/.cache}/nanochat-turk}"
export TRAIN_SHARDS="${TRAIN_SHARDS:--1}"
export DEPTH="${DEPTH:-21}"
export VOCAB_SIZE="${VOCAB_SIZE:-32768}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-bpe_${VOCAB_SIZE}}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
export RUN_CETVEL="${RUN_CETVEL:-1}"
export CETVEL_SUITE="${CETVEL_SUITE:-core}"

bash runs/turkish_foundation.sh
