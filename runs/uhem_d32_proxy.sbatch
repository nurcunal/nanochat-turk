#!/bin/bash

# Pack the proxy arms four-at-a-time on exactly one 4xA100 node.
# Submit with PROXY_PHASE=screen first, then PROXY_PHASE=confirm after screening.

#SBATCH --job-name=tr-d32-wd-proxy
#SBATCH --account=nakane
#SBATCH --partition=a100x4q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:4
#SBATCH --time=2-00:00:00
#SBATCH --mem=900G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v1.json}"
PREFLIGHT_RECEIPT="${PREFLIGHT_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/preflight.json}"
ATTENTION_PROBE_RECEIPT="${ATTENTION_PROBE_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/attention_probe.json}"
SCREENING_RECEIPT="${SCREENING_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/proxy_screening.json}"
WD_PROXY_APPROVAL="${WD_PROXY_APPROVAL:-$NANOCHAT_BASE_DIR/control/d32/wd_proxy_approval.json}"
PROXY_PHASE="${PROXY_PHASE:-}"
export CODE_DIR NANOCHAT_BASE_DIR RECIPE PREFLIGHT_RECEIPT ATTENTION_PROBE_RECEIPT

cd "$CODE_DIR"
if [ "$PROXY_PHASE" != screen ] && [ "$PROXY_PHASE" != confirm ]; then
    echo "PROXY_PHASE must be exactly screen or confirm" >&2
    exit 2
fi
if [ "${SLURM_JOB_PARTITION:-}" != a100x4q ]; then
    echo "Proxy packing is authorized only on a100x4q; no single-GPU fallback" >&2
    exit 2
fi
gpu_count="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
if [ "$gpu_count" -ne 4 ]; then
    echo "Proxy allocation must expose exactly four A100 GPUs, found $gpu_count" >&2
    exit 2
fi
.venv/bin/python scripts/d32_family_workflow.py validate-recipe --recipe="$RECIPE"
test -f "$PREFLIGHT_RECEIPT"
test -f "$ATTENTION_PROBE_RECEIPT"

run_packed_arms() {
    local depth="$1"
    shift
    local arms=("$@")
    local cursor=0
    while [ "$cursor" -lt "${#arms[@]}" ]; do
        local pids=()
        local labels=()
        local slot
        for slot in 0 1 2 3; do
            if [ "$cursor" -ge "${#arms[@]}" ]; then
                break
            fi
            read -r candidate seed <<< "${arms[$cursor]}"
            echo "Launching packed proxy arm depth=$depth candidate=$candidate seed=$seed gpu=$slot"
            bash runs/uhem_d32_proxy_arm.sh "$depth" "$candidate" "$seed" "$slot" &
            pids+=("$!")
            labels+=("$candidate/$seed")
            cursor=$((cursor + 1))
        done
        local failed=0
        local index
        for index in "${!pids[@]}"; do
            if ! wait "${pids[$index]}"; then
                echo "Proxy arm failed: ${labels[$index]}" >&2
                failed=1
            fi
        done
        if [ "$failed" -ne 0 ]; then
            exit 1
        fi
    done
}

candidates=(
    upstream_92d63d4e_control
    legacy_transferred
    half_transferred_constant_cooldown_wd
    half_transferred_linear_cooldown_wd
    no_weight_decay
)

if [ "$PROXY_PHASE" = screen ]; then
    arms=()
    receipt_args=()
    for candidate in "${candidates[@]}"; do
        for seed in 42 314159; do
            arms+=("$candidate $seed")
            run_id="tr_d32_general_bpe32k_v1_proxy_d12_${candidate}_seed${seed}"
            receipt_args+=(--run-receipt="$NANOCHAT_BASE_DIR/control/d32/proxy_runs/${run_id}.json")
        done
    done
    run_packed_arms 12 "${arms[@]}"
    .venv/bin/python scripts/d32_family_workflow.py screen-proxy \
        --recipe="$RECIPE" \
        --preflight-receipt="$PREFLIGHT_RECEIPT" \
        --attention-probe="$ATTENTION_PROBE_RECEIPT" \
        "${receipt_args[@]}" \
        --output="$SCREENING_RECEIPT"
else
    test -f "$SCREENING_RECEIPT"
    mapfile -t advanced_candidates < <(
        .venv/bin/python -c 'import json,sys; x=json.load(open(sys.argv[1])); [print(v) for v in x["advanced_candidate_ids"]]' "$SCREENING_RECEIPT"
    )
    if [ "${#advanced_candidates[@]}" -ne 2 ]; then
        echo "Screening receipt must advance exactly two WSD candidates" >&2
        exit 2
    fi
    confirmation_candidates=(upstream_92d63d4e_control "${advanced_candidates[@]}")
    arms=()
    receipt_args=()
    for candidate in "${confirmation_candidates[@]}"; do
        arms+=("$candidate 42")
        run_id="tr_d32_general_bpe32k_v1_proxy_d20_${candidate}_seed42"
        receipt_args+=(--run-receipt="$NANOCHAT_BASE_DIR/control/d32/proxy_runs/${run_id}.json")
    done
    run_packed_arms 20 "${arms[@]}"
    .venv/bin/python scripts/d32_family_workflow.py accept-proxy \
        --recipe="$RECIPE" \
        --preflight-receipt="$PREFLIGHT_RECEIPT" \
        --attention-probe="$ATTENTION_PROBE_RECEIPT" \
        --screening-receipt="$SCREENING_RECEIPT" \
        "${receipt_args[@]}" \
        --output="$WD_PROXY_APPROVAL"
fi
