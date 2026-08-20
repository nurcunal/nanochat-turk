#!/bin/bash

# Explicit, fail-closed submitter. With no argument it only prints the plan.

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

mode="${1:---plan}"
case "$mode" in
    --plan|--submit-static-launcher-probe|--submit-proxy-chain|--submit-signal-resume-gate|--submit-smoke-chain) ;;
    *)
        echo "usage: $0 [--plan|--submit-static-launcher-probe|--submit-proxy-chain|--submit-signal-resume-gate|--submit-smoke-chain]" >&2
        exit 2
        ;;
esac

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v1.json}"
PREFLIGHT_RECEIPT="${PREFLIGHT_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/preflight.json}"
ATTENTION_PROBE_RECEIPT="${ATTENTION_PROBE_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/attention_probe.json}"
WD_PROXY_APPROVAL="${WD_PROXY_APPROVAL:-$NANOCHAT_BASE_DIR/control/d32/wd_proxy_approval.json}"
STATIC_LAUNCHER_GATE="${STATIC_LAUNCHER_GATE:-$NANOCHAT_BASE_DIR/control/d32/static_launcher_gate_ws4.json}"
SIGNAL_RESUME_GATE="${SIGNAL_RESUME_GATE:-$NANOCHAT_BASE_DIR/control/d32/signal_resume_gate_ws4.json}"
PRODUCTION_GATE="${PRODUCTION_GATE:-$NANOCHAT_BASE_DIR/control/d32/production_topology_gate.json}"

cd "$CODE_DIR"
.venv/bin/python scripts/d32_family_workflow.py validate-recipe --recipe="$RECIPE"

if [ "$mode" = --plan ]; then
    echo "No job will be submitted in --plan mode."
    echo "Required order: data-prep resource/storage sample -> corpus/tokenizer -> family preflight -> 4-rank static launcher gate -> 1xA100 attention probe -> packed d12 proxy -> packed d20 confirmation -> bounded signal/requeue/resume gate -> 8-GPU smoke -> optional 16-GPU smoke -> topology selection."
    for artifact in "$PREFLIGHT_RECEIPT" "$STATIC_LAUNCHER_GATE" "$ATTENTION_PROBE_RECEIPT" "$WD_PROXY_APPROVAL" "$SIGNAL_RESUME_GATE" "$PRODUCTION_GATE"; do
        if [ -f "$artifact" ]; then
            echo "present: $artifact"
        else
            echo "missing: $artifact"
        fi
    done
    upstream=92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
    if git merge-base --is-ancestor "$upstream" HEAD; then
        echo "pinned-upstream ancestry: PASS"
    else
        echo "pinned-upstream ancestry: BLOCKED (HEAD is not descended from $upstream)"
    fi
    if [ -n "$(git status --porcelain)" ]; then
        echo "clean-worktree preflight: BLOCKED"
    else
        echo "clean-worktree preflight: PASS"
    fi
    echo "Proxy chain command: $0 --submit-proxy-chain"
    echo "Static launcher probe command: $0 --submit-static-launcher-probe"
    echo "Signal/requeue/resume gate command (only after proxy acceptance): $0 --submit-signal-resume-gate"
    echo "Smoke chain command (only after proxy acceptance): $0 --submit-smoke-chain"
    echo "Production submission is deliberately unavailable until review and every gate are complete."
    exit 0
fi

common_export="ALL,CODE_DIR=$CODE_DIR,NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR,RECIPE=$RECIPE,PREFLIGHT_RECEIPT=$PREFLIGHT_RECEIPT,ATTENTION_PROBE_RECEIPT=$ATTENTION_PROBE_RECEIPT,WSD_PROXY_APPROVAL=$WD_PROXY_APPROVAL,STATIC_LAUNCHER_GATE=$STATIC_LAUNCHER_GATE,SIGNAL_RESUME_GATE=$SIGNAL_RESUME_GATE"

if [ "$mode" = --submit-static-launcher-probe ]; then
    test -f "$PREFLIGHT_RECEIPT"
    probe_job="$(sbatch --parsable --export="$common_export" runs/uhem_d32_static_launcher_probe.sbatch)"
    final_job="$(sbatch --parsable --dependency="afterok:$probe_job" --export="$common_export,PROBE_JOB_ID=$probe_job" runs/uhem_d32_static_launcher_finalize.sbatch)"
    echo "static_launcher_probe_job=$probe_job"
    echo "static_launcher_finalize_job=$final_job"
    exit 0
fi

if [ "$mode" = --submit-proxy-chain ]; then
    test -f "$PREFLIGHT_RECEIPT"
    if [ -f "$ATTENTION_PROBE_RECEIPT" ]; then
        probe_dependency=""
    else
        probe_job="$(sbatch --parsable --export="$common_export" runs/uhem_d32_attention_probe.sbatch)"
        probe_dependency="--dependency=afterok:$probe_job"
        echo "attention_probe_job=$probe_job"
    fi
    screen_job="$(sbatch --parsable ${probe_dependency:+"$probe_dependency"} --export="$common_export,PROXY_PHASE=screen" runs/uhem_d32_proxy.sbatch)"
    confirm_job="$(sbatch --parsable --dependency="afterok:$screen_job" --export="$common_export,PROXY_PHASE=confirm" runs/uhem_d32_proxy.sbatch)"
    echo "proxy_screen_job=$screen_job"
    echo "proxy_confirmation_job=$confirm_job"
    exit 0
fi

if [ "$mode" = --submit-signal-resume-gate ]; then
    test -f "$PREFLIGHT_RECEIPT"
    test -f "$ATTENTION_PROBE_RECEIPT"
    test -f "$WD_PROXY_APPROVAL"
    test -f "$STATIC_LAUNCHER_GATE"
    test ! -e "$SIGNAL_RESUME_GATE"
    signal_job="$(sbatch --parsable --export="$common_export" runs/uhem_d32_signal_resume_smoke.sbatch)"
    signal_final="$(sbatch --parsable --dependency="afterok:$signal_job" --export="$common_export,SIGNAL_JOB_ID=$signal_job" runs/uhem_d32_signal_resume_finalize.sbatch)"
    echo "signal_resume_smoke_job=$signal_job"
    echo "signal_resume_finalize_job=$signal_final"
    exit 0
fi

if [ "$mode" = --submit-smoke-chain ]; then
    test -f "$PREFLIGHT_RECEIPT"
    test -f "$ATTENTION_PROBE_RECEIPT"
    test -f "$WD_PROXY_APPROVAL"
    test -f "$STATIC_LAUNCHER_GATE"
    test -f "$SIGNAL_RESUME_GATE"
    smoke8_job="$(sbatch --parsable --nodes=2 --export="$common_export" runs/uhem_d32_smoke.sbatch)"
    smoke8_final="$(sbatch --parsable --dependency="afterok:$smoke8_job" --export="$common_export,SMOKE_JOB_ID=$smoke8_job,SMOKE_NODES=2,STATIC_LAUNCHER_GATE=$STATIC_LAUNCHER_GATE" runs/uhem_d32_smoke_finalize.sbatch)"
    smoke16_job="$(sbatch --parsable --nodes=4 --dependency="afterok:$smoke8_final" --export="$common_export,STATIC_LAUNCHER_GATE=$STATIC_LAUNCHER_GATE" runs/uhem_d32_smoke.sbatch)"
    smoke16_final="$(sbatch --parsable --dependency="afterok:$smoke16_job" --export="$common_export,SMOKE_JOB_ID=$smoke16_job,SMOKE_NODES=4,STATIC_LAUNCHER_GATE=$STATIC_LAUNCHER_GATE" runs/uhem_d32_smoke_finalize.sbatch)"
    gate_job="$(sbatch --parsable --dependency="afterok:$smoke16_final" --export="$common_export" runs/uhem_d32_smoke_gate.sbatch)"
    echo "smoke_8gpu_job=$smoke8_job"
    echo "smoke_8gpu_finalize_job=$smoke8_final"
    echo "smoke_16gpu_job=$smoke16_job"
    echo "smoke_16gpu_finalize_job=$smoke16_final"
    echo "production_gate_job=$gate_job"
    exit 0
fi
