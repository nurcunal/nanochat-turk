from __future__ import annotations

import re
from pathlib import Path

from scripts.build_turkish_pretrain_corpus import build_parser as build_corpus_parser
from scripts.d32_wsd_train import build_parser
from scripts.upload_base_checkpoint_to_hf import _family_model_card


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "runs" / "uhem_d32_train_node.sh"


def test_shared_wrapper_supplies_every_required_trainer_option() -> None:
    """Catch argparse failures before any A100 allocation is requested."""

    source = WRAPPER.read_text(encoding="utf-8")
    parser = build_parser()
    required = {
        option
        for action in parser._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }
    supplied = set(re.findall(r"--[a-z0-9][a-z0-9-]+", source))
    assert not required - supplied
    assert 'if [ "$RUN_KIND" = production ]' in source
    assert 'train_args+=("--production-gate=$PRODUCTION_GATE")' in source
    assert '"--preflight-receipt=$PREFLIGHT_RECEIPT"' in source
    assert 'train_args+=("--packing-capacity-receipt=$PACKING_CAPACITY_RECEIPT")' in source
    assert 'expected_capacity_receipt="$DATA_DIR/packing_capacity_receipt.json"' in source
    assert '--launcher="$LAUNCHER_ID"' in source


def test_proxy_and_srun_paths_use_truthful_launcher_identities() -> None:
    proxy = (ROOT / "runs" / "uhem_d32_proxy_arm.sh").read_text(encoding="utf-8")
    assert "LAUNCHER_ID=slurm_batch_direct_python_env_v1" in proxy
    for relative in (
        "runs/uhem_d32_production.sbatch",
        "runs/uhem_d32_signal_resume_smoke.sbatch",
        "runs/uhem_d32_smoke.sbatch",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "LAUNCHER_ID=slurm_srun_direct_python_env_v1" in source


def test_every_wrapper_mode_exports_preflight_and_uses_dedicated_trainer() -> None:
    launchers = {
        "production": ROOT / "runs" / "uhem_d32_production.sbatch",
        "smoke": ROOT / "runs" / "uhem_d32_smoke.sbatch",
        "signal_smoke": ROOT / "runs" / "uhem_d32_signal_resume_smoke.sbatch",
        "proxy": ROOT / "runs" / "uhem_d32_proxy_arm.sh",
    }
    wrapper_source = WRAPPER.read_text(encoding="utf-8")
    assert "-m scripts.d32_wsd_train" in wrapper_source
    assert re.search(r"(?m)^[^#\n]*\b(?:torchrun|mpirun)\b", wrapper_source) is None
    for run_kind, path in launchers.items():
        source = path.read_text(encoding="utf-8")
        assert "PREFLIGHT_RECEIPT" in source, (run_kind, path)
        assert "uhem_d32_train_node.sh" in source, (run_kind, path)
        if run_kind != "proxy":
            assert f"RUN_KIND={run_kind}" in source, (run_kind, path)
    production = launchers["production"].read_text(encoding="utf-8")
    assert "export PRODUCTION_GATE=" in production


def test_distributed_launchers_are_slurm_20_static_srun_only() -> None:
    launchers = (
        ROOT / "runs" / "uhem_d32_production.sbatch",
        ROOT / "runs" / "uhem_d32_smoke.sbatch",
        ROOT / "runs" / "uhem_d32_signal_resume_smoke.sbatch",
        ROOT / "runs" / "uhem_d32_static_launcher_probe.sbatch",
    )
    required = (
        "srun ",
        "--ntasks-per-node=4",
        "--gpus-per-task=1",
        "--gpu-bind=single:1",
        "--kill-on-bad-exit=1",
    )
    for path in launchers:
        source = path.read_text(encoding="utf-8")
        assert all(fragment in source for fragment in required), path
        assert re.search(r"(?m)^[^#\n]*\b(?:torchrun|mpirun)\b", source) is None, path


def test_static_probe_uses_module_mode_for_repo_imports() -> None:
    for relative in (
        "runs/uhem_d32_static_launcher_probe.sbatch",
        "runs/uhem_d32_smoke.sbatch",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert ".venv/bin/python -m scripts.d32_static_launch_probe" in source
        assert ".venv/bin/python scripts/d32_static_launch_probe.py" not in source


def test_training_environment_setup_is_frozen_and_version_pinned() -> None:
    source = (ROOT / "runs" / "uhem_d32_prepare_training_env.sh").read_text(
        encoding="utf-8"
    )
    assert "expected_uv_version=0.11.29" in source
    assert '"$UV_BIN" sync --frozen --extra gpu' in source
    assert "uv sync --locked" not in source
    assert "c91fdd03ae9705565572eee31924d4c0bca24bf5431a8eabff4c061882f94929" in source
    assert "de7891b832854162111208644ddb72685069ce8128e2bed9dbb7993aa6af5861" in source


def test_family_upload_uses_module_mode_for_repo_imports() -> None:
    source = (ROOT / "runs" / "uhem_d32_family_upload.sbatch").read_text(
        encoding="utf-8"
    )
    assert ".venv/bin/python -m scripts.upload_base_checkpoint_to_hf" in source
    assert ".venv/bin/python scripts/upload_base_checkpoint_to_hf.py" not in source


def test_family_upload_retains_end_to_end_reproduction_sources() -> None:
    source = (ROOT / "scripts" / "upload_base_checkpoint_to_hf.py").read_text(
        encoding="utf-8"
    )
    required = (
        'repo_root / "pyproject.toml"',
        '"scripts/build_turkish_pretrain_corpus.py"',
        '"scripts/audit_turkish_backend_sample.py"',
        '"scripts/train_turkish_raw_bpe.py"',
        '"scripts/turkish_packed_sample.py"',
        '"scripts/turkish_packed_production.py"',
        '"scripts/d32_static_launch_probe.py"',
        '"scripts/upload_base_checkpoint_to_hf.py"',
        '"nanochat/turkish_backend.py"',
        '"nanochat/turkish_corpus.py"',
        '"nanochat/tokenizer_quality.py"',
        '"configs/pretrain/fineweb2_tur_Latn.yml"',
        '"configs/pretrain/glotlid_calibration_tr_v1.jsonl"',
        '"environments/turkish-data/uv.lock"',
        '"schemas/artifact-manifest.schema.json"',
        '"schemas/dataset-manifest.schema.json"',
        '"runs/uhem_d32_prepare_training_env.sh"',
        '"runs/uhem_d32_data_prep_storage_sample.sbatch"',
        '"runs/uhem_d32_data_prep_writer_probe.sbatch"',
        '"runs/uhem_turkish_data_objects.sbatch"',
        '"runs/uhem_turkish_data_objects_packed_sample.sbatch"',
        '"runs/uhem_turkish_data_objects_packed_production.sbatch"',
        '"runs/uhem_turkish_data_buckets_packed_sample.sbatch"',
        '"runs/uhem_turkish_data_buckets_packed_production.sbatch"',
        '"runs/uhem_turkish_sample_quality_audit.sbatch"',
        '"runs/uhem_turkish_data_bootstrap.sbatch"',
        '"runs/uhem_turkish_prepare_data_env.sbatch"',
        '"runs/uhem_turkish_packing_preflight.sbatch"',
        '"runs/uhem_turkish_production_pool.sbatch"',
        '"runs/uhem_turkish_tokenizer_sample.sbatch"',
        '"runs/uhem_turkish_tokenizer_train.sbatch"',
        '"runs/uhem_turkish_tokenizer_quality.sbatch"',
    )
    assert all(path in source for path in required)


def test_family_upload_includes_tokenizer_sample_and_quality_evidence() -> None:
    source = (ROOT / "scripts" / "upload_base_checkpoint_to_hf.py").read_text(
        encoding="utf-8"
    )
    assert "validate_tokenizer_quality_gate" in source
    assert '("quality_report.json", "quality_approval.json")' in source
    assert 'sample_root / "tokenizer_sample_manifest.json"' in source
    assert 'sample_root / "fineweb2_manifest.json"' in source
    assert 'training_receipt.get("sample_manifest_sha256") != sample_sha' in source
    for evidence in (
        "parent_pool_manifest.json",
        "parent_pool_ownership.json",
        "qa_report.json",
        "packing_preflight_report.json",
        "packing_preflight_approval.json",
        "cluster-launch-receipt",
    ):
        assert evidence in source


def test_production_data_and_tokenizer_launchers_consume_exact_gate() -> None:
    contracts = {
        "runs/uhem_turkish_data_objects_packed_production.sbatch": (
            "#SBATCH --time=2-00:00:00",
            "#SBATCH --ntasks=32",
            "#SBATCH --cpus-per-task=4",
        ),
        "runs/uhem_turkish_data_buckets_packed_production.sbatch": (
            "#SBATCH --time=1-00:00:00",
            "#SBATCH --ntasks=14",
            "#SBATCH --cpus-per-task=8",
        ),
        "runs/uhem_turkish_production_pool.sbatch": (
            "#SBATCH --time=2-00:00:00",
            "--cluster-launch-receipt",
            "production-pool",
        ),
        "runs/uhem_turkish_tokenizer_sample.sbatch": (
            "#SBATCH --time=12:00:00",
            "--pool-dir",
            "--cluster-launch-receipt",
        ),
        "runs/uhem_turkish_tokenizer_train.sbatch": (
            "#SBATCH --time=1-00:00:00",
            "--tokenizer-sample-dir",
            'BASELINE_TOKENIZER_DIR:-/ari/users/nunal/nanochat-turk-d20-bpe32k/tokenizers/bpe_32768',
            '--baseline-tokenizer-dir "$BASELINE_TOKENIZER_DIR"',
            "--baseline-preflight-only",
            'test -f "$TOKENIZER_DIR/tokenizer.tiktoken"',
            "--cluster-launch-receipt",
        ),
        "runs/uhem_turkish_tokenizer_quality.sbatch": (
            "#SBATCH --time=12:00:00",
            "--tokenizer-quality-dir",
            "--cluster-launch-receipt",
        ),
        "runs/uhem_turkish_packing_preflight.sbatch": (
            "#SBATCH --time=12:00:00",
            "--pool-dir",
            "--cluster-launch-receipt",
        ),
        "runs/uhem_turkish_corpus_finalize.sbatch": (
            "#SBATCH --time=2-00:00:00",
            "--packing-preflight-dir",
            "--cluster-launch-receipt",
        ),
    }
    for relative, required in contracts.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "scripts.turkish_packed_production validate-gate" in source
        assert all(fragment in source for fragment in required), relative
        if relative == "runs/uhem_turkish_tokenizer_train.sbatch":
            assert source.index("--baseline-preflight-only") < source.index(
                'mkdir -p "$TOKENIZER_DIR" "$TOKENIZER_QUALITY_DIR"'
            )
    for relative in (
        "runs/uhem_turkish_data_objects_packed_production.sbatch",
        "runs/uhem_turkish_data_buckets_packed_production.sbatch",
        "runs/uhem_turkish_data_cluster.sbatch",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert '--write-dir="$DATA_RUN_DIR"' in source, relative


def test_operator_runbook_names_all_data_and_tokenizer_gates() -> None:
    source = (ROOT / "docs" / "tr_d32_turkish_wsd_uhem.md").read_text(
        encoding="utf-8"
    )
    for fragment in (
        "runs/uhem_turkish_sample_quality_audit.sbatch",
        "manual mixture-quality approval",
        "resource approval",
        "PRODUCTION_DATA_NODES",
        "bash runs/uhem_d32_prepare_training_env.sh",
        "BASELINE_TOKENIZER_DIR",
        "tokenizer.tiktoken",
        "APPROVED_SOURCE_TOKENS",
        "QUOTA_HEADROOM_BYTES",
        "Any mixture weight, source selector, or accepted-source policy change invalidates",
    ):
        assert fragment in source


def test_family_preflight_and_upload_require_exact_cluster_launch() -> None:
    workflow_source = (ROOT / "scripts" / "d32_family_workflow.py").read_text(
        encoding="utf-8"
    )
    upload_source = (ROOT / "scripts" / "upload_base_checkpoint_to_hf.py").read_text(
        encoding="utf-8"
    )
    upload_wrapper = (ROOT / "runs" / "uhem_d32_family_upload.sbatch").read_text(
        encoding="utf-8"
    )
    assert 'preflight.add_argument("--cluster-launch-receipt"' in workflow_source
    assert '"production_cluster_launch_receipt_sha256": cluster_launch_sha' in workflow_source
    assert 'parser.add_argument("--cluster-launch-receipt"' in upload_source
    assert 'expected_chain.get("cluster_launch_receipt_sha256")' in upload_source
    assert '--cluster-launch-receipt="$CLUSTER_LAUNCH_RECEIPT"' in upload_wrapper


def test_v2_exposure_index_requires_and_launches_with_family_identity() -> None:
    parser = build_corpus_parser()
    index_parser = next(
        action
        for action in parser._actions
        if action.dest == "command"
    ).choices["exposure-index"]
    family_action = next(
        action for action in index_parser._actions if action.dest == "family_id"
    )
    assert family_action.required is True

    finalize = (ROOT / "runs" / "uhem_turkish_corpus_finalize.sbatch").read_text(
        encoding="utf-8"
    )
    assert 'FAMILY_RECIPE="${FAMILY_RECIPE:-' in finalize
    assert 'exposure-index \\' in finalize
    assert '--family-id "$FAMILY_ID"' in finalize
    assert '--study-manifest-sha256 "$FAMILY_RECIPE_SHA256"' in finalize


def test_family_upload_binds_macocu_and_uses_recipe_tokenizer_name() -> None:
    source = (ROOT / "scripts" / "upload_base_checkpoint_to_hf.py").read_text(
        encoding="utf-8"
    )
    assert 'source_receipt_sha != preflight["corpus"].get(' in source
    assert 'source_macocu.get("manifest_sha256") != manifest_sha' in source
    assert 'preflight_macocu.get("sha256") != manifest_sha' in source
    assert "The tokenizer is `{recipe['artifacts']['tokenizer_name']}`" in source

    card = _family_model_card(
        {
            "artifacts": {"tokenizer_name": "fixture_dynamic_tokenizer"},
            "checkpoints": {"finals": []},
        },
        "omit",
        {"family_recipe_sha256": "a" * 64, "code_revision": "b" * 40},
    )
    assert "`fixture_dynamic_tokenizer`" in card


def test_turkish_data_environment_setup_is_frozen_and_version_pinned() -> None:
    source = (ROOT / "runs" / "uhem_turkish_prepare_data_env.sbatch").read_text(
        encoding="utf-8"
    )
    assert '"$UV_BIN" sync --project "$PROJECT_DIR" --locked' in source
    assert "--reinstall-package fasttext-numpy2-wheel" in source
    assert '!= 0.11.29' in source
    assert "7ba5aecacc2720a71307c3e033baf53feeb382cf9eabb483146ca504a2ecce63" in source
    assert "7c53680fb9f85fbb117f29e611bb2a08294f3b539426b72df3295bc7959010a7" in source
    assert 'metadata.version("fasttext-numpy2-wheel") == "0.9.2"' in source
    assert 'metadata.version("spacy") == "3.8.15"' in source
    assert 'metadata.version("fasttext-wheel")' in source
    assert "legacy fasttext-wheel must not be installed" in source
    assert 'assert "np.asarray(probs)" in predict_source' in source
    project = (ROOT / "environments" / "turkish-data" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    lock = (ROOT / "environments" / "turkish-data" / "uv.lock").read_text(
        encoding="utf-8"
    )
    assert '"fasttext-numpy2-wheel==0.9.2"' in project
    assert '\nname = "fasttext-wheel"\n' not in lock


def test_uhem_module_initialization_precedes_bash_nounset() -> None:
    launchers = sorted((ROOT / "runs").glob("uhem_d32_*")) + sorted(
        (ROOT / "runs").glob("uhem_turkish_*")
    )
    checked = 0
    for path in launchers:
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        marker = "source /etc/profile.d/modules.sh"
        if marker not in source:
            continue
        checked += 1
        assert source.index(marker) < source.index("\nset -u\n"), path
        assert source.index(marker) > source.index("set -eo pipefail"), path
    assert checked >= 21


def test_turkish_data_bootstrap_pins_and_seals_prerequisites() -> None:
    source = (ROOT / "runs" / "uhem_turkish_data_bootstrap.sbatch").read_text(
        encoding="utf-8"
    )
    assert "scripts/turkish_data_backend.py resolve" in source
    assert "scripts/turkish_data_backend.py fetch-glotlid" in source
    assert "scripts/turkish_data_backend.py calibrate" in source
    assert "scripts/turkish_data_backend.py sample-ranks" in source
    assert "--locked" in source
    assert 'export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"' in source
    assert 'mv "$sample_tmp" "$SAMPLE_RANKS"' in source


def test_turkish_data_jobs_use_the_pinned_uv_binary() -> None:
    for relative in (
        "runs/uhem_turkish_data_objects.sbatch",
        "runs/uhem_turkish_data_buckets.sbatch",
        "runs/uhem_turkish_data_cluster.sbatch",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert 'UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"' in source
        assert '"$UV_BIN" run --project environments/turkish-data --locked' in source


def test_data_prep_storage_bridge_is_dependent_and_pre_safety() -> None:
    storage_source = (
        ROOT / "runs" / "uhem_d32_data_prep_storage_sample.sbatch"
    ).read_text(encoding="utf-8")
    writer_source = (
        ROOT / "runs" / "uhem_d32_data_prep_writer_probe.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=128" in storage_source
    assert "SAMPLE_BUCKET_JOB_ID" in storage_source
    assert "SAMPLE_BUCKET_LAUNCH_RECEIPT" in storage_source
    assert "seal-data-prep-pack-plan" in storage_source
    assert "seal-data-prep-storage-sample" in storage_source
    assert "data-prep-storage-gate" in storage_source
    assert '"$MACOCU_MANIFEST" "$BACKEND_RESOURCE_REPORT" "$WRITER_PROBE"' in storage_source
    assert 'test -f "$input"' in storage_source
    assert "scripts/turkish_data_backend.py resource-report" not in storage_source
    assert 'test ! -e "$STORAGE_SAMPLE"' in storage_source
    assert 'test ! -e "$STORAGE_GATE"' in storage_source

    assert "#SBATCH --cpus-per-task=128" in writer_source
    assert '--billable-cpus-per-job 128 --safety-factor 1' in writer_source
    assert "resource-report" in writer_source
    assert "seal-data-prep-writer-probe" in writer_source
    assert 'scratch/data_prep_writer_probe/job${SLURM_JOB_ID:?}' in writer_source
    assert 'test "$BASE_DEVICE" = "$SCRATCH_DEVICE"' in writer_source
    assert 'test ! -e "$BACKEND_RESOURCE_REPORT"' in writer_source


def test_signal_path_targets_the_srun_step_on_slurm_20() -> None:
    production = (ROOT / "runs" / "uhem_d32_production.sbatch").read_text(
        encoding="utf-8"
    )
    smoke = (ROOT / "runs" / "uhem_d32_signal_resume_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --signal=USR1@900" in production
    assert "#SBATCH --signal=B:" not in production
    assert "#SBATCH --signal=USR1@900" in smoke
    assert "#SBATCH --signal=B:" not in smoke
    assert 'scancel --signal=USR1 "${SLURM_JOB_ID}.0"' in smoke
    assert "scontrol signal_job" not in smoke
