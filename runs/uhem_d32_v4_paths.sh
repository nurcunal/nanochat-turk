#!/bin/bash

# Canonical BeeGFS artifact namespace for the Turkish d32 v4 family.
# Source this file after NANOCHAT_BASE_DIR is set. A non-empty override must
# equal the canonical path byte-for-byte, so a v4 job cannot accidentally
# write into a historical lineage or a second alias tree.

if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
    echo "NANOCHAT_BASE_DIR must be set before sourcing uhem_d32_v4_paths.sh" >&2
    return 2 2>/dev/null || exit 2
fi

D32_V4_CONTROL_DIR="$NANOCHAT_BASE_DIR/control/data_v4"
D32_V4_FAMILY_CONTROL_DIR="$NANOCHAT_BASE_DIR/control/d32_v4"
D32_V4_SAMPLE_RUN_DIR="$NANOCHAT_BASE_DIR/data_backend/resource_sample_v4"
D32_V4_DATA_RUN_DIR="$NANOCHAT_BASE_DIR/data_backend/production_v4"
D32_V4_POOL_DIR="$NANOCHAT_BASE_DIR/data_v4/filtered_pool"
D32_V4_TOKENIZER_SAMPLE_DIR="$NANOCHAT_BASE_DIR/control/tokenizer/tr_general_raw_bpe_32k_v4/sample"
D32_V4_TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizers/tr_general_raw_bpe_32k_v4"
D32_V4_TOKENIZER_QUALITY_DIR="$NANOCHAT_BASE_DIR/control/tokenizer/tr_general_raw_bpe_32k_v4/quality"
D32_V4_PACKING_CONTROL_DIR="$NANOCHAT_BASE_DIR/control/packing/tr_general_clean_v4"
D32_V4_FINAL_CORPUS_DIR="$NANOCHAT_BASE_DIR/pretrain_data/tr_general_clean_v4"
D32_V4_SOURCE_PLAN="$D32_V4_CONTROL_DIR/source_plan.json"
D32_V4_CALIBRATION="$D32_V4_CONTROL_DIR/backend_calibration.json"
D32_V4_SAMPLE_RANKS="$D32_V4_CONTROL_DIR/resource_sample_ranks.json"
D32_V4_LANE_PLAN="$D32_V4_CONTROL_DIR/resource_sample_lane_plan.json"
D32_V4_AUDIT_OUTPUT_DIR="$D32_V4_CONTROL_DIR/sample_quality_audit"
D32_V4_BACKEND_RESOURCE_REPORT="$D32_V4_CONTROL_DIR/backend_resource_report.json"
D32_V4_WRITER_PROBE="$D32_V4_CONTROL_DIR/post_cluster_writer_probe.json"
D32_V4_MIXTURE_QUALITY_APPROVAL="$D32_V4_CONTROL_DIR/mixture_quality_approval.json"
D32_V4_RESOURCE_APPROVAL="$D32_V4_CONTROL_DIR/resource_approval.json"
D32_V4_PRODUCTION_NODE_SELECTION="$D32_V4_CONTROL_DIR/production_data_node_selection.json"
D32_V4_PACK_PLAN="$D32_V4_CONTROL_DIR/production_source_pack_plan.json"
D32_V4_STORAGE_SAMPLE="$D32_V4_CONTROL_DIR/d32_data_prep_storage_sample.json"
D32_V4_DATA_PREP_STORAGE_GATE="$D32_V4_FAMILY_CONTROL_DIR/data_prep_storage_gate.json"
D32_V4_SOURCE_RECEIPT="$D32_V4_CONTROL_DIR/source_receipt.json"
D32_V4_BACKEND_RECEIPT="$D32_V4_CONTROL_DIR/backend_receipt.json"

d32_v4_bind_canonical_path() {
    if [ "$#" -ne 2 ]; then
        echo "d32_v4_bind_canonical_path requires VARIABLE and CANONICAL_PATH" >&2
        return 2
    fi
    local variable_name="$1"
    local canonical_path="$2"
    case "$variable_name" in
        CONTROL_DIR|FAMILY_CONTROL_DIR|SAMPLE_RUN_DIR|DATA_RUN_DIR|POOL_DIR|TOKENIZER_SAMPLE_DIR|TOKENIZER_DIR|TOKENIZER_QUALITY_DIR|PACKING_CONTROL_DIR|FINAL_CORPUS_DIR|SOURCE_PLAN|CALIBRATION|SAMPLE_RANKS|LANE_PLAN|SAMPLE_LANE_PLAN|AUDIT_OUTPUT_DIR|BACKEND_RESOURCE_REPORT|WRITER_PROBE|MIXTURE_QUALITY_APPROVAL|RESOURCE_APPROVAL|PRODUCTION_NODE_SELECTION|PACK_PLAN|PRODUCTION_PACK_PLAN|STORAGE_SAMPLE|DATA_PREP_STORAGE_GATE|STORAGE_GATE|SOURCE_RECEIPT|BACKEND_RECEIPT) ;;
        *)
            echo "unsupported d32 v4 path variable: $variable_name" >&2
            return 2
            ;;
    esac
    local supplied_path="${!variable_name:-}"
    if [ -n "$supplied_path" ] && [ "$supplied_path" != "$canonical_path" ]; then
        echo "$variable_name must equal the frozen d32 v4 path: $canonical_path" >&2
        return 2
    fi
    printf -v "$variable_name" '%s' "$canonical_path"
    export "$variable_name"
}

D32_CONTROL_DIR="$D32_V4_CONTROL_DIR"
D32_FAMILY_CONTROL_DIR="$D32_V4_FAMILY_CONTROL_DIR"
D32_SAMPLE_RUN_DIR="$D32_V4_SAMPLE_RUN_DIR"
D32_DATA_RUN_DIR="$D32_V4_DATA_RUN_DIR"
D32_POOL_DIR="$D32_V4_POOL_DIR"
D32_TOKENIZER_SAMPLE_DIR="$D32_V4_TOKENIZER_SAMPLE_DIR"
D32_TOKENIZER_DIR="$D32_V4_TOKENIZER_DIR"
D32_TOKENIZER_QUALITY_DIR="$D32_V4_TOKENIZER_QUALITY_DIR"
D32_PACKING_CONTROL_DIR="$D32_V4_PACKING_CONTROL_DIR"
D32_FINAL_CORPUS_DIR="$D32_V4_FINAL_CORPUS_DIR"
D32_SOURCE_PLAN="$D32_V4_SOURCE_PLAN"
D32_CALIBRATION="$D32_V4_CALIBRATION"
D32_SAMPLE_RANKS="$D32_V4_SAMPLE_RANKS"
D32_LANE_PLAN="$D32_V4_LANE_PLAN"
D32_AUDIT_OUTPUT_DIR="$D32_V4_AUDIT_OUTPUT_DIR"
D32_BACKEND_RESOURCE_REPORT="$D32_V4_BACKEND_RESOURCE_REPORT"
D32_WRITER_PROBE="$D32_V4_WRITER_PROBE"
D32_MIXTURE_QUALITY_APPROVAL="$D32_V4_MIXTURE_QUALITY_APPROVAL"
D32_RESOURCE_APPROVAL="$D32_V4_RESOURCE_APPROVAL"
D32_PRODUCTION_NODE_SELECTION="$D32_V4_PRODUCTION_NODE_SELECTION"
D32_PACK_PLAN="$D32_V4_PACK_PLAN"
D32_STORAGE_SAMPLE="$D32_V4_STORAGE_SAMPLE"
D32_DATA_PREP_STORAGE_GATE="$D32_V4_DATA_PREP_STORAGE_GATE"
D32_SOURCE_RECEIPT="$D32_V4_SOURCE_RECEIPT"
D32_BACKEND_RECEIPT="$D32_V4_BACKEND_RECEIPT"

d32_bind_canonical_path() {
    d32_v4_bind_canonical_path "$@"
}
