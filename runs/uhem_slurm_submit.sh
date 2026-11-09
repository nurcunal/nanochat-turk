#!/bin/bash

# UHeM's sbatch wrapper may print billing notices to stdout even with
# --parsable. Keep those diagnostics visible, but return exactly one validated
# numeric allocation ID to command substitution. Some wrapper failures also
# return status zero, so absence of an ID is always fatal.

d32_submit_sbatch() {
    if [ "$#" -eq 0 ]; then
        echo "d32_submit_sbatch requires sbatch arguments" >&2
        return 2
    fi

    local submission_output submission_rc line candidate job_id=""
    if submission_output="$(sbatch --parsable "$@" 2>&1)"; then
        submission_rc=0
    else
        submission_rc=$?
    fi
    printf '%s\n' "$submission_output" >&2
    if [ "$submission_rc" -ne 0 ]; then
        return "$submission_rc"
    fi

    while IFS= read -r line; do
        candidate="${line%%;*}"
        case "$candidate" in
            ''|*[!0-9]*) continue ;;
        esac
        if [ -n "$job_id" ] && [ "$job_id" != "$candidate" ]; then
            echo "sbatch returned multiple different allocation IDs" >&2
            return 2
        fi
        job_id="$candidate"
    done <<< "$submission_output"

    if [ -z "$job_id" ]; then
        echo "sbatch returned success without a parsable allocation ID" >&2
        return 2
    fi
    printf '%s\n' "$job_id"
}
