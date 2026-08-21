from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nanochat.experiment_manifest import load_json_strict, seal_manifest, write_json_atomic
from scripts import d32_data_prep_operator as operator


ROOT = Path(__file__).resolve().parents[1]
PATH_CONTRACT = ROOT / "runs" / "uhem_d32_v3_paths.sh"
PATH_CONTRACT_V4 = ROOT / "runs" / "uhem_d32_v4_paths.sh"
CLEAN_CODE_CONTRACT = ROOT / "runs" / "uhem_d32_require_clean_code.sh"


def _shell(source: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", source, "d32-v3-path-test", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_v3_path_contract_binds_the_exact_artifact_namespace(tmp_path: Path) -> None:
    result = _shell(
        r'''
set -eu
NANOCHAT_BASE_DIR="$1"
source "$2"
d32_v3_bind_canonical_path POOL_DIR "$D32_V3_POOL_DIR"
d32_v3_bind_canonical_path TOKENIZER_SAMPLE_DIR "$D32_V3_TOKENIZER_SAMPLE_DIR"
d32_v3_bind_canonical_path TOKENIZER_DIR "$D32_V3_TOKENIZER_DIR"
d32_v3_bind_canonical_path TOKENIZER_QUALITY_DIR "$D32_V3_TOKENIZER_QUALITY_DIR"
d32_v3_bind_canonical_path PACKING_CONTROL_DIR "$D32_V3_PACKING_CONTROL_DIR"
d32_v3_bind_canonical_path FINAL_CORPUS_DIR "$D32_V3_FINAL_CORPUS_DIR"
printf '%s\n' "$POOL_DIR" "$TOKENIZER_SAMPLE_DIR" "$TOKENIZER_DIR" \
  "$TOKENIZER_QUALITY_DIR" "$PACKING_CONTROL_DIR" "$FINAL_CORPUS_DIR"
''',
        str(tmp_path),
        str(PATH_CONTRACT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(tmp_path / "data_v3" / "filtered_pool"),
        str(
            tmp_path
            / "control"
            / "tokenizer"
            / "tr_general_raw_bpe_32k_v3"
            / "sample"
        ),
        str(tmp_path / "tokenizers" / "tr_general_raw_bpe_32k_v3"),
        str(
            tmp_path
            / "control"
            / "tokenizer"
            / "tr_general_raw_bpe_32k_v3"
            / "quality"
        ),
        str(tmp_path / "control" / "packing" / "tr_general_clean_v3"),
        str(tmp_path / "pretrain_data" / "tr_general_clean_v3"),
    ]


def test_v3_path_contract_rejects_a_redirected_output(tmp_path: Path) -> None:
    result = _shell(
        r'''
set -eu
NANOCHAT_BASE_DIR="$1"
POOL_DIR="$1/stale-v2-pool"
source "$2"
d32_v3_bind_canonical_path POOL_DIR "$D32_V3_POOL_DIR"
''',
        str(tmp_path),
        str(PATH_CONTRACT),
    )
    assert result.returncode == 2
    assert "must equal the frozen d32 v3 path" in result.stderr


def test_v4_path_contract_binds_the_exact_artifact_namespace(tmp_path: Path) -> None:
    result = _shell(
        r'''
set -eu
NANOCHAT_BASE_DIR="$1"
source "$2"
for variable in CONTROL_DIR SAMPLE_RUN_DIR DATA_RUN_DIR POOL_DIR \
  TOKENIZER_SAMPLE_DIR TOKENIZER_DIR TOKENIZER_QUALITY_DIR \
  PACKING_CONTROL_DIR FINAL_CORPUS_DIR; do
  canonical_variable="D32_${variable}"
  d32_bind_canonical_path "$variable" "${!canonical_variable}"
done
printf '%s\n' "$CONTROL_DIR" "$SAMPLE_RUN_DIR" "$DATA_RUN_DIR" "$POOL_DIR" \
  "$TOKENIZER_SAMPLE_DIR" "$TOKENIZER_DIR" "$TOKENIZER_QUALITY_DIR" \
  "$PACKING_CONTROL_DIR" "$FINAL_CORPUS_DIR"
''',
        str(tmp_path),
        str(PATH_CONTRACT_V4),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(tmp_path / "control" / "data_v4"),
        str(tmp_path / "data_backend" / "resource_sample_v4"),
        str(tmp_path / "data_backend" / "production_v4"),
        str(tmp_path / "data_v4" / "filtered_pool"),
        str(tmp_path / "control" / "tokenizer" / "tr_general_raw_bpe_32k_v4" / "sample"),
        str(tmp_path / "tokenizers" / "tr_general_raw_bpe_32k_v4"),
        str(tmp_path / "control" / "tokenizer" / "tr_general_raw_bpe_32k_v4" / "quality"),
        str(tmp_path / "control" / "packing" / "tr_general_clean_v4"),
        str(tmp_path / "pretrain_data" / "tr_general_clean_v4"),
    ]


def _init_git_fixture(root: Path) -> str:
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    (root / "tracked.txt").write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def test_clean_code_contract_allows_old_scalar_and_current_array_logs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head = _init_git_fixture(repo)
    (repo / "older-job-123.out").write_text("log\n", encoding="utf-8")
    (repo / "active-array-456_7.err").write_text("log\n", encoding="utf-8")
    result = _shell(
        r'''
set -eu
CODE_REVISION="$3"
SLURM_JOB_NAME=active-array
SLURM_JOB_ID=456
SLURM_ARRAY_JOB_ID=456
SLURM_ARRAY_TASK_ID=7
source "$2"
d32_require_clean_committed_code "$1"
printf '%s\n' "$CODE_REVISION"
''',
        str(repo), str(CLEAN_CODE_CONTRACT), head,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == head


def test_clean_code_contract_rejects_untracked_code(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_fixture(repo)
    (repo / "uncommitted.py").write_text("raise SystemExit\n", encoding="utf-8")
    result = _shell(
        r'''
set -eu
source "$2"
d32_require_clean_committed_code "$1"
''',
        str(repo), str(CLEAN_CODE_CONTRACT),
    )
    assert result.returncode == 2
    assert "worktree is not clean" in result.stderr


def test_node_selector_is_deterministic_and_fails_when_none_pass() -> None:
    assert (
        operator._select_first_passing(
            [
                {"node_count": 4, "passes_gate_limits": True},
                {"node_count": 1, "passes_gate_limits": False},
                {"node_count": 2, "passes_gate_limits": True},
            ]
        )
        == 2
    )
    with pytest.raises(
        operator.workflow.FamilyWorkflowError,
        match="no production data node count passes",
    ):
        operator._select_first_passing(
            [{"node_count": 1, "passes_gate_limits": False}]
        )


def test_node_selection_command_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluations = [
        {"node_count": 1, "passes_gate_limits": False},
        {"node_count": 2, "passes_gate_limits": True},
    ]
    bindings = {"family_id": "fixture", "recipe_sha256": "a" * 64}
    monkeypatch.setattr(
        operator, "_node_evaluations", lambda _args: (bindings, evaluations)
    )
    output = tmp_path / "selection.json"
    argv = [
        "select-production-nodes",
        "--source-plan",
        "plan.json",
        "--calibration",
        "calibration.json",
        "--sample-run-dir",
        "sample",
        "--backend-resource-report",
        "report.json",
        "--writer-probe",
        "writer.json",
        "--output",
        str(output),
    ]
    assert operator.main(argv) == 0
    assert capsys.readouterr().out.strip() == "2"
    receipt = load_json_strict(output)
    assert receipt["selected_nodes"] == 2
    assert receipt["maximum_evaluated_nodes"] == 2
    assert operator.main(argv) == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_node_selection_validator_recomputes_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluations = [
        {"node_count": 1, "passes_gate_limits": False},
        {"node_count": 2, "passes_gate_limits": True},
    ]
    bindings = {"family_id": "fixture", "recipe_sha256": "a" * 64}
    monkeypatch.setattr(
        operator, "_node_evaluations", lambda _args: (bindings, evaluations)
    )
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_data_prep_production_node_selection",
            **bindings,
            "selection_policy": (
                "smallest_positive_node_count_passing_exact_existing_"
                "storage_gate_walltime_and_rss_arithmetic"
            ),
            "selected_nodes": 2,
            "maximum_evaluated_nodes": 2,
            "evaluations": evaluations,
            "created_at_utc": "2026-08-21T00:00:00+00:00",
            "canonical_sha256": None,
        }
    )
    valid = tmp_path / "valid.json"
    write_json_atomic(valid, receipt)
    base = [
        "validate-production-nodes",
        "--source-plan",
        "plan.json",
        "--calibration",
        "calibration.json",
        "--sample-run-dir",
        "sample",
        "--backend-resource-report",
        "report.json",
        "--writer-probe",
        "writer.json",
        "--node-selection",
    ]
    assert operator.main([*base, str(valid)]) == 0
    assert capsys.readouterr().out.strip() == "2"

    drifted = dict(receipt)
    drifted["selected_nodes"] = 1
    drifted["canonical_sha256"] = None
    drifted = seal_manifest(drifted)
    drifted_path = tmp_path / "drifted.json"
    write_json_atomic(drifted_path, drifted)
    assert operator.main([*base, str(drifted_path)]) == 2
    assert "drifted from current evidence" in capsys.readouterr().err


def test_live_headroom_command_uses_recipe_identity_and_rejects_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = {
        "storage": {"uhem_live_quota": {"uid": 4500, "storage_pool_id": 1}}
    }
    monkeypatch.setattr(operator.workflow, "load_recipe", lambda _path: (recipe, "x"))
    calls = []

    def fake_live(_root, *, uid: int, storage_pool_id: int, path: Path):
        calls.append((uid, storage_pool_id, path))
        return 123456, {"effective_free_bytes": 123456}

    monkeypatch.setattr(operator.workflow, "_live_beegfs_storage", fake_live)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    assert (
        operator.main(
            [
                "live-beegfs-headroom",
                "--recipe",
                "recipe.json",
                "--work-dir",
                str(work_dir),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "123456"
    assert calls == [(4500, 1, work_dir.resolve())]

    link = tmp_path / "work-link"
    link.symlink_to(work_dir, target_is_directory=True)
    assert (
        operator.main(
            [
                "live-beegfs-headroom",
                "--recipe",
                "recipe.json",
                "--work-dir",
                str(link),
            ]
        )
        == 2
    )
    assert "not be a symlink" in capsys.readouterr().err
    assert len(calls) == 1
