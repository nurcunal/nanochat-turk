#!/bin/bash

# CETVEL assessment for the completed FineWeb-2 Turkish d32/raw-BPE model.
#
# Modes:
#   smoke      - two examples each from Belebele and corrected XQuAD
#   core11     - historical classification protocol, directly comparable to
#                the June 2026 tokenizer-ablation table
#   xquad_d32  - corrected XQuAD generation for the new d32 checkpoint
#   xquad_d20  - corrected XQuAD generation for the raw-BPE d20 reference
#   full20_tail - the remaining eight tasks needed to complete CETVEL full-20;
#                 combine with core11 and corrected d32 XQuAD

#SBATCH --job-name=tr-d32-fw2-cetvel
#SBATCH --account=nakane
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --output=/ari/users/nunal/nanochat-turk-d32-fineweb2-bpe32k/launch/%x-%j.out
#SBATCH --error=/ari/users/nunal/nanochat-turk-d32-fineweb2-bpe32k/launch/%x-%j.err

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
unset PYTHONPATH
set -euo pipefail

MODE="${MODE:-core11}"
case "$MODE" in
    smoke|core11|xquad_d32|xquad_d20|full20_tail) ;;
    *)
        echo "MODE must be smoke, core11, xquad_d32, xquad_d20, or full20_tail" >&2
        exit 2
        ;;
esac

export ARTIFACT_BASE_DIR="${ARTIFACT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-fineweb2-bpe32k}"
export EVAL_CODE_DIR="${EVAL_CODE_DIR:-/ari/users/nunal/nanochat-turk}"
export CETVEL_DIR="${CETVEL_DIR:-/ari/users/nunal/nanochat-turk-d20-bpe32k/cetvel}"
export FIXED_GENERATION_RUNNER="${FIXED_GENERATION_RUNNER:-$ARTIFACT_BASE_DIR/launch/cetvel_generation_fixed_runner.py}"
export NANOCHAT_TOKENIZER_NAME=bpe_32768
export HF_HOME="${HF_HOME:-/ari/users/nunal/nanochat-turk-d20-bpe32k/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/ari/users/nunal/nanochat-turk-d20-bpe32k/cetvel_hf_datasets_cache_datasets219}"
export WANDB_PROJECT=nanochat-turk
export WANDB_DIR="$ARTIFACT_BASE_DIR/wandb"
export WANDB_CONSOLE=redirect
export WANDB_CONSOLE_MULTIPART=true
export WANDB_CONSOLE_CHUNK_MAX_SECONDS=60
export PYTHONPATH="$EVAL_CODE_DIR:$CETVEL_DIR/lm-evaluation-harness"
export PYTHONUNBUFFERED=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CETVEL_GENERATION_PROGRESS=1
export CETVEL_GENERATION_PROGRESS_EVERY=1

expected_adapter_sha256=47d53a044fc6920d8b51fd187166c73dbe107c344c28e68d250d6074ef71733c
expected_evaluator_sha256=55a587c64f419a24d1dafe3943dfa583f7525b44b9cab26224f792c1f36da15a
expected_runner_sha256=338ab274ccb67ba3ad4129629a70cbd629b0b8dee11386d133961281dae2aded

if [ ! -x "$EVAL_CODE_DIR/.venv/bin/python" ]; then
    echo "Missing historical CETVEL environment: $EVAL_CODE_DIR/.venv" >&2
    exit 2
fi
if [ ! -d "$CETVEL_DIR/lm-evaluation-harness" ]; then
    echo "Missing pinned CETVEL checkout: $CETVEL_DIR" >&2
    exit 2
fi
actual_adapter_sha256="$(sha256sum "$EVAL_CODE_DIR/nanochat/lm_eval_nanochat.py" | awk '{print $1}')"
actual_evaluator_sha256="$(sha256sum "$EVAL_CODE_DIR/scripts/cetvel_eval.py" | awk '{print $1}')"
if [ "$actual_adapter_sha256" != "$expected_adapter_sha256" ] || \
   [ "$actual_evaluator_sha256" != "$expected_evaluator_sha256" ]; then
    echo "Historical CETVEL evaluator drifted" >&2
    exit 2
fi
if [ ! -f "$FIXED_GENERATION_RUNNER" ]; then
    echo "Missing corrected generation runner: $FIXED_GENERATION_RUNNER" >&2
    exit 2
fi
actual_runner_sha256="$(sha256sum "$FIXED_GENERATION_RUNNER" | awk '{print $1}')"
if [ "$actual_runner_sha256" != "$expected_runner_sha256" ]; then
    echo "Corrected generation runner drifted" >&2
    exit 2
fi

case "$MODE" in
    xquad_d20)
        export NANOCHAT_BASE_DIR=/ari/users/nunal/nanochat-turk-d20-bpe32k
        export MODEL_TAG=tr_d20_bpe_32768_chinchilla20
        export CETVEL_MODEL_STEP=17100
        expected_depth=20
        ;;
    *)
        export NANOCHAT_BASE_DIR="$ARTIFACT_BASE_DIR"
        export MODEL_TAG=tr_d32_bpe_32768_fineweb2_all_v1
        export CETVEL_MODEL_STEP=28000
        expected_depth=32
        ;;
esac

step_tag="$(printf '%06d' "$CETVEL_MODEL_STEP")"
checkpoint_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
model_path="$checkpoint_dir/model_${step_tag}.pt"
meta_path="$checkpoint_dir/meta_${step_tag}.json"
if [ ! -f "$model_path" ] || [ ! -f "$meta_path" ]; then
    echo "Missing checkpoint at step $CETVEL_MODEL_STEP in $checkpoint_dir" >&2
    exit 2
fi
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME/tokenizer.pkl" ]; then
    echo "Missing tokenizer $NANOCHAT_TOKENIZER_NAME under $NANOCHAT_BASE_DIR" >&2
    exit 2
fi

"$EVAL_CODE_DIR/.venv/bin/python" - "$meta_path" "$expected_depth" <<'PY'
import json
import sys

meta = json.load(open(sys.argv[1], encoding="utf-8"))
config = meta["model_config"]
expected = {
    "n_layer": int(sys.argv[2]),
    "sequence_len": 2048,
    "vocab_size": 32768,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"checkpoint model_config mismatch for {key}: "
            f"expected {value}, observed {config.get(key)}"
        )
print("CETVEL checkpoint preflight OK", expected)
PY

"$EVAL_CODE_DIR/.venv/bin/python" - <<'PY'
import importlib.metadata as metadata
import sys

expected = {
    "datasets": "2.19.2",
    "lm-eval": "0.4.2",
    "torch": "2.9.1+cu128",
    "wandb": "0.21.3",
}
for package, version in expected.items():
    observed = metadata.version(package)
    if observed != version:
        raise SystemExit(
            f"historical CETVEL environment drifted for {package}: "
            f"expected {version}, observed {observed}"
        )
print("CETVEL runtime preflight OK", sys.version.split()[0], expected)
PY

core11_tasks="exams_tr,belebele_tr,turkish_plu,cetvel_xcopa_tr,cetvel_xnli_tr,mnli_tr,snli_tr,news_cat,offenseval_tr,trclaim19,xfact_tr"
full20_tail_tasks="tquad,mkqa_tr,wmt-tr-en-prompt,wmt-en-tr-prompt,mlsum_tr,xlsum_tr,wiki_lingua_tr,gecturk_generation"
suite=core
if [ "$MODE" = "full20_tail" ]; then
    suite=full
fi
args=(
    "--suite=$suite"
    "--model-tag=$MODEL_TAG"
    "--model-step=$CETVEL_MODEL_STEP"
    --batch-size=1
    --device=cuda:0
    --max-gen-tokens=128
    --task-progress
)
use_fixed_runner=0

case "$MODE" in
    smoke)
        use_fixed_runner=1
        args+=(
            --tasks=belebele_tr,xquad_tr
            --limit=2
            "--output-path=$ARTIFACT_BASE_DIR/cetvel_out/smoke_job${SLURM_JOB_ID}"
        )
        ;;
    core11)
        export WANDB_RUN=cetvel-d32-bpe32k-fineweb2-all-core11-legacy-v1
        export WANDB_RUN_ID=cetveld32bpe32kfw2core11legacyv1
        export WANDB_RESUME=allow
        args+=(
            "--tasks=$core11_tasks"
            "--output-path=$ARTIFACT_BASE_DIR/cetvel_out/core11_legacy_step028000_v1"
            --wandb
        )
        ;;
    xquad_d32)
        use_fixed_runner=1
        export WANDB_RUN=cetvel-d32-bpe32k-fineweb2-all-xquad-fixed-v1
        export WANDB_RUN_ID=cetveld32bpe32kfw2xquadfixedv1
        export WANDB_RESUME=allow
        args+=(
            --tasks=xquad_tr
            "--output-path=$ARTIFACT_BASE_DIR/cetvel_out/xquad_fixed_step028000_v1"
            --wandb
        )
        ;;
    xquad_d20)
        use_fixed_runner=1
        export WANDB_RUN=cetvel-d20-bpe32k-xquad-fixed-reference-v1
        export WANDB_RUN_ID=cetveld20bpe32kxquadfixedrefv1
        export WANDB_RESUME=allow
        args+=(
            --tasks=xquad_tr
            "--output-path=$ARTIFACT_BASE_DIR/cetvel_out/reference_d20_xquad_fixed_step017100_v1"
            --wandb
        )
        ;;
    full20_tail)
        use_fixed_runner=1
        export WANDB_RUN=cetvel-d32-bpe32k-fineweb2-all-full20-tail-fixed-v1
        export WANDB_RUN_ID=cetveld32bpe32kfw2full20tailfixedv1
        export WANDB_RESUME=allow
        args+=(
            "--tasks=$full20_tail_tasks"
            "--output-path=$ARTIFACT_BASE_DIR/cetvel_out/full20_tail_fixed_step028000_v1"
            --wandb
        )
        ;;
esac

echo "MODE=$MODE"
echo "HOST=$(hostname)"
echo "MODEL_TAG=$MODEL_TAG MODEL_STEP=$CETVEL_MODEL_STEP"
echo "ADAPTER_SHA256=$actual_adapter_sha256"
echo "EVALUATOR_SHA256=$actual_evaluator_sha256"
echo "RUNNER_SHA256=$actual_runner_sha256"
echo "CETVEL_ARGS=${args[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$use_fixed_runner" -eq 1 ]; then
    export CETVEL_LEGACY_EVAL_SCRIPT="$EVAL_CODE_DIR/scripts/cetvel_eval.py"
    export CETVEL_FORCE_SDPA=1
    exec "$EVAL_CODE_DIR/.venv/bin/python" "$FIXED_GENERATION_RUNNER" "${args[@]}"
fi
exec "$EVAL_CODE_DIR/.venv/bin/python" "$EVAL_CODE_DIR/scripts/cetvel_eval.py" "${args[@]}"
