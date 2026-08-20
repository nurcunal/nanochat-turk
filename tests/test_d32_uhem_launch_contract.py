from __future__ import annotations

import re
from pathlib import Path

from scripts.d32_wsd_train import build_parser


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
    assert "uv sync --frozen --extra gpu" in source
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
        '"scripts/train_turkish_raw_bpe.py"',
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
        '"runs/uhem_turkish_data_objects.sbatch"',
        '"runs/uhem_turkish_packing_preflight.sbatch"',
    )
    assert all(path in source for path in required)


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
