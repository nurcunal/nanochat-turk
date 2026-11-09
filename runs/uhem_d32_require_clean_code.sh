#!/bin/bash

# Fail before producing lineage artifacts unless CODE_DIR is an exact clean
# checkout. The only tolerated untracked paths are root-level Slurm-shaped
# stdout/stderr files from current or completed allocations; the runbook routes
# new logs outside CODE_DIR, while this allowance keeps older scheduler logs
# from making an otherwise exact checkout unusable.

d32_require_clean_committed_code() {
    if [ "$#" -ne 1 ]; then
        echo "d32_require_clean_committed_code requires CODE_DIR" >&2
        return 2
    fi
    local requested_root="$1"
    local physical_root git_root head status line path
    physical_root="$(cd "$requested_root" && pwd -P)" || return 2
    git_root="$(git -C "$physical_root" rev-parse --show-toplevel 2>/dev/null)" || {
        echo "CODE_DIR is not a Git checkout: $physical_root" >&2
        return 2
    }
    git_root="$(cd "$git_root" && pwd -P)" || return 2
    if [ "$git_root" != "$physical_root" ]; then
        echo "CODE_DIR must be the Git worktree root: $git_root" >&2
        return 2
    fi
    head="$(git -C "$git_root" rev-parse --verify HEAD 2>/dev/null)" || {
        echo "cannot resolve the committed code revision" >&2
        return 2
    }
    case "$head" in
        *[!0-9a-f]*|'')
            echo "committed code revision is not a full hexadecimal object ID" >&2
            return 2
            ;;
    esac
    if [ "${#head}" -ne 40 ] && [ "${#head}" -ne 64 ]; then
        echo "committed code revision must be a full Git object ID" >&2
        return 2
    fi
    if [ -n "${CODE_REVISION:-}" ] && [ "$CODE_REVISION" != "$head" ]; then
        echo "CODE_REVISION does not equal checked-out HEAD: $head" >&2
        return 2
    fi
    if ! git -C "$git_root" diff --quiet --ignore-submodules -- ||
       ! git -C "$git_root" diff --cached --quiet --ignore-submodules --; then
        echo "data-prep code worktree has staged or unstaged tracked changes" >&2
        return 2
    fi
    status="$(git -C "$git_root" status --porcelain=v1 --untracked-files=all)"
    if [ -n "$status" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] || continue
            case "$line" in
                "?? ${SLURM_JOB_NAME:-__no_job__}-${SLURM_JOB_ID:-__no_id__}.out"|\
                "?? ${SLURM_JOB_NAME:-__no_job__}-${SLURM_JOB_ID:-__no_id__}.err")
                    continue
                    ;;
            esac
            if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] &&
               [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
                case "$line" in
                    "?? ${SLURM_JOB_NAME:-__no_job__}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"|\
                    "?? ${SLURM_JOB_NAME:-__no_job__}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err")
                        continue
                        ;;
                esac
            fi
            path="${line#?? }"
            # Older allocations may have left root-level scheduler logs before
            # the runbook began routing logs outside CODE_DIR. They are inert
            # and cannot shadow Python/config inputs; nested paths and every
            # other untracked filename still fail.
            if [[ "$line" == "?? "* ]] && [[ "$path" != */* ]] &&
               [[ "$path" =~ ^[A-Za-z0-9._+-]+-[0-9]+(_[0-9]+)?\.(out|err)$ ]]; then
                continue
            fi
            echo "data-prep code worktree is not clean: $path" >&2
            return 2
        done <<< "$status"
    fi
    CODE_REVISION="$head"
    export CODE_REVISION
}
