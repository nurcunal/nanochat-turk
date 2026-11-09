"""Pure fail-closed contracts for the dedicated Turkish WSD trainer."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from nanochat.experiment_manifest import (
    canonical_json,
    load_json_strict,
    validate_dataset_manifest,
    verify_manifest_hash,
)
from nanochat.exposure import (
    validate_exposure_manifest,
    validate_training_exposure_plan,
)
from nanochat.wsd import validate_weight_decay_proxy_candidates


PINNED_UPSTREAM_REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
FAMILY_ID = "tr_d32_general_bpe32k_v1"  # Backwards-compatible v1 alias.
FAMILY_ID_V2 = "tr_d32_general_bpe32k_v2"
FAMILY_ID_V3 = "tr_d32_general_bpe32k_v3"
FAMILY_ID_V4 = "tr_d32_general_bpe32k_v4"

_REPEAT_CAPACITY_FAMILY_IDS = frozenset({FAMILY_ID_V3, FAMILY_ID_V4})

_TRAINING_EXPOSURE_ARTIFACTS = {
    "proxy_d12_seed42_ws1": "training_exposure_proxy_d12_seed42_ws1.json",
    "proxy_d12_seed314159_ws1": (
        "training_exposure_proxy_d12_seed314159_ws1.json"
    ),
    "proxy_d20_seed42_ws1": "training_exposure_proxy_d20_seed42_ws1.json",
    "signal_smoke_ws4_seed42": "training_exposure_signal_smoke_ws4_seed42.json",
    "smoke_ws8": "training_exposure_smoke_ws8.json",
    "smoke_ws16": "training_exposure_smoke_ws16.json",
    "trunk_ws8_seed42": "training_exposure_trunk_ws8_seed42.json",
    "s12_ws8_seed42": "training_exposure_s12_ws8_seed42.json",
    "s20_ws8_seed42": "training_exposure_s20_ws8_seed42.json",
    "s40_ws8_seed42": "training_exposure_s40_ws8_seed42.json",
    "trunk_ws16_seed42": "training_exposure_trunk_ws16_seed42.json",
    "s12_ws16_seed42": "training_exposure_s12_ws16_seed42.json",
    "s20_ws16_seed42": "training_exposure_s20_ws16_seed42.json",
    "s40_ws16_seed42": "training_exposure_s40_ws16_seed42.json",
}


def _artifact_contract(*, version: str) -> dict[str, Any]:
    corpus_id = f"tr_general_clean_{version}"
    artifacts: dict[str, Any] = {
        "mixture_config": f"configs/pretrain/tr_d32_turkish_general_{version}.json",
        "corpus_id": corpus_id,
        "corpus_root": f"pretrain_data/{corpus_id}",
        "corpus_manifest": "corpus_manifest.json",
        "nanochat_dataset_manifest": "fineweb2_manifest.json",
        "source_receipt": "source_receipt.json",
        "packing_capacity_receipt": "packing_capacity_receipt.json",
        "validation_exposure_manifest": "validation_exposure_manifest.json",
        "exposure_plan_index": "exposure_plan_index.json",
        "training_exposure_manifests": dict(_TRAINING_EXPOSURE_ARTIFACTS),
        "tokenizer_name": f"tr_general_raw_bpe_32k_{version}",
        "tokenizer_root": f"tokenizers/tr_general_raw_bpe_32k_{version}",
        "tokenizer_package_manifest": "package_manifest.json",
    }
    if version in {"v2", "v3", "v4"}:
        artifacts["macocu_preparation_manifest"] = (
            "source_data/macocu_genre_tr_v1/manifest.json"
        )
    if version in {"v3", "v4"}:
        artifacts.update(
            {
                "mot_preparation_manifest": (
                    "source_data/mot_tr_v1_11/manifest.json"
                ),
                "parlamint_preparation_manifest": (
                    "source_data/parlamint_tr_v5_0/manifest.json"
                ),
            }
        )
    return artifacts


FAMILY_ARTIFACT_CONTRACTS = {
    FAMILY_ID: _artifact_contract(version="v1"),
    FAMILY_ID_V2: _artifact_contract(version="v2"),
    FAMILY_ID_V3: _artifact_contract(version="v3"),
    FAMILY_ID_V4: _artifact_contract(version="v4"),
}


def family_artifact_contract(family_id: str) -> dict[str, Any]:
    """Return the one exact artifact namespace permitted for a family."""

    expected = FAMILY_ARTIFACT_CONTRACTS.get(family_id)
    if expected is None:
        raise StrictTrainingError("unexpected d32 family recipe identity")
    return {
        **expected,
        "training_exposure_manifests": dict(
            expected["training_exposure_manifests"]
        ),
    }


PREEMPTION_EXIT_CODE = 75


class StrictTrainingError(ValueError):
    """Raised before compute when a production invariant is not satisfied."""


@dataclass(frozen=True)
class SeedPlan:
    study_seed: int
    rank: int
    model_init: int
    runtime: int

    def to_dict(self) -> dict[str, int]:
        return {
            "study_seed": self.study_seed,
            "rank": self.rank,
            "model_init": self.model_init,
            "runtime": self.runtime,
        }


@dataclass(frozen=True)
class ArtifactBindings:
    recipe: dict[str, Any]
    recipe_sha256: str
    dataset_manifest: dict[str, Any]
    dataset_sha256: str
    exposure_plan: dict[str, Any]
    exposure_plan_sha256: str
    validation_manifest: dict[str, Any]
    validation_sha256: str


def _validate_anchor_preflight_binding(
    path: Path,
    *,
    source_id: str,
    derived_sources: Mapping[str, Any],
    preflight_record: Any,
) -> None:
    """Revalidate an accepted native-text preparation before GPU construction."""

    if path.name != "manifest.json" or path.is_symlink() or not path.is_file():
        raise StrictTrainingError(
            f"{source_id} preparation manifest is missing or unsafe"
        )
    if not isinstance(preflight_record, Mapping):
        raise StrictTrainingError(
            f"preflight lacks the {source_id} preparation manifest"
        )
    try:
        from nanochat.turkish_anchor_preparation import validate_anchor_preparation

        manifest = validate_anchor_preparation(path.parent, verify_files=True)
        digest = verify_manifest_hash(manifest)
    except (OSError, ValueError) as exc:
        raise StrictTrainingError(
            f"invalid accepted preparation for {source_id}: {exc}"
        ) from exc
    production = manifest.get("production_acceptance")
    artifacts = manifest.get("artifacts")
    acquisition = manifest.get("acquisition_receipt")
    if not all(
        isinstance(value, Mapping)
        for value in (production, artifacts, acquisition)
    ):
        raise StrictTrainingError(f"{source_id} preparation provenance is malformed")
    acceptance = production.get("receipt")
    data = artifacts.get("data")
    if not isinstance(acceptance, Mapping) or not isinstance(data, Mapping):
        raise StrictTrainingError(f"{source_id} preparation evidence is malformed")
    expected_provenance = {
        "manifest_uri": path.resolve().as_uri(),
        "manifest_sha256": digest,
        "source_id": source_id,
        "preparer_version": manifest.get("preparer_version"),
        "production_acceptance": {
            "stage": "accepted_production",
            "receipt_sha256": acceptance.get("canonical_sha256"),
        },
        "acquisition_receipt_sha256": acquisition.get("canonical_sha256"),
        "clean": manifest.get("clean"),
        "data_artifact": {
            "logical_jsonl_sha256": data.get("logical_jsonl_sha256"),
            "totals": data.get("totals"),
        },
        "downstream_admission": {
            "preparer_automatically_admits_training": False,
            "backend_turkish_no_code_audit_required": True,
        },
    }
    if (
        manifest.get("source_id") != source_id
        or derived_sources.get(source_id) != expected_provenance
        or Path(str(preflight_record.get("path", ""))).resolve() != path.resolve()
        or preflight_record.get("sha256") != digest
    ):
        raise StrictTrainingError(
            f"{source_id} preparation/source/preflight binding drifted"
        )


def derive_seed_plan(study_seed: int, *, rank: int = 0) -> SeedPlan:
    """Record literal upstream-compatible seeding; do not invent rank streams.

    Pinned Nanochat initializes every rank with literal seed 42.  The strict
    lane generalizes that only by using the literal requested study seed (42
    for production/control; a second literal seed for a declared proxy arm).
    The best-fit loader is deterministic and GPT has no dropout, so there is no
    separate data-order or dropout seed to claim.
    """

    if isinstance(study_seed, bool) or not isinstance(study_seed, int) or study_seed < 0:
        raise StrictTrainingError("seed must be a non-negative integer")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise StrictTrainingError("rank must be a non-negative integer")
    return SeedPlan(
        study_seed=study_seed,
        rank=rank,
        model_init=study_seed,
        runtime=study_seed,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StrictTrainingError(f"cannot verify Git provenance: {exc}") from exc


def verify_code_provenance(
    repo_root: str | Path,
    *,
    expected_revision: str,
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the exact clean deployment commit and pinned upstream core."""

    root = Path(repo_root).resolve()
    if not isinstance(expected_revision, str) or not expected_revision:
        raise StrictTrainingError("--code-revision must be a full Git object ID")
    head = _git(root, "rev-parse", "HEAD")
    if head != expected_revision:
        raise StrictTrainingError("working-tree HEAD differs from --code-revision")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise StrictTrainingError("strict training requires a clean Git worktree")
    code = recipe.get("code_provenance")
    if not isinstance(code, Mapping):
        raise StrictTrainingError("family recipe has no code_provenance object")
    if code.get("upstream_base_revision") != PINNED_UPSTREAM_REVISION:
        raise StrictTrainingError("family recipe upstream revision drifted")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", PINNED_UPSTREAM_REVISION, head],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StrictTrainingError("deployment commit is not a pinned-upstream descendant") from exc
    exact = code.get("exact_file_sha256")
    if not isinstance(exact, Mapping) or not exact:
        raise StrictTrainingError("family recipe has no exact core hash inventory")
    verified: dict[str, str] = {}
    for relative_path, expected_hash in sorted(exact.items()):
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise StrictTrainingError("core hash inventory is malformed")
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise StrictTrainingError("core hash path escapes repository") from exc
        actual_hash = file_sha256(candidate)
        if actual_hash != expected_hash:
            raise StrictTrainingError(f"pinned core file hash drifted: {relative_path}")
        verified[relative_path] = actual_hash
    lock_path = root / "uv.lock"
    pyproject_path = root / "pyproject.toml"
    environment = code.get("training_environment")
    if not isinstance(environment, Mapping):
        raise StrictTrainingError("family recipe has no training environment contract")
    if not pyproject_path.is_file() or not lock_path.is_file():
        raise StrictTrainingError("pyproject.toml or uv.lock is missing")
    pyproject_sha = file_sha256(pyproject_path)
    lock_sha = file_sha256(lock_path)
    if pyproject_sha != environment.get("pyproject_sha256"):
        raise StrictTrainingError("pinned training pyproject.toml drifted")
    if lock_sha != environment.get("uv_lock_sha256"):
        raise StrictTrainingError("pinned training uv.lock drifted")
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    if actual_python != environment.get("python_version"):
        raise StrictTrainingError("training Python differs from the pinned environment")
    return {
        "git_revision": head,
        "upstream_base_revision": PINNED_UPSTREAM_REVISION,
        "environment_lock_sha256": lock_sha,
        "environment_pyproject_sha256": pyproject_sha,
        "python_version": actual_python,
        "exact_core_sha256": verified,
    }


def validate_family_recipe(recipe: Mapping[str, Any]) -> None:
    """Validate the safety-critical, family-wide recipe declarations."""

    family_id = recipe.get("family_id")
    if recipe.get("schema_version") != "1.0" or family_id not in FAMILY_ARTIFACT_CONTRACTS:
        raise StrictTrainingError("unexpected d32 family recipe identity")
    artifacts = recipe.get("artifacts")
    if not isinstance(artifacts, Mapping) or dict(artifacts) != family_artifact_contract(
        str(family_id)
    ):
        raise StrictTrainingError(
            "family recipe artifacts differ from the exact family identity"
        )
    language = recipe.get("language_policy")
    if not isinstance(language, Mapping) or language.get("allowed_languages") != ["tr"]:
        raise StrictTrainingError("family recipe must be Turkish-only")
    for field in (
        "allow_code_corpora",
        "allow_synthetic_instruction_data",
        "allow_translated_filler",
    ):
        if language.get(field) is not False:
            raise StrictTrainingError(f"language_policy.{field} must be false")
    code = recipe.get("code_provenance")
    expected_environment = {
        "manager": "uv",
        "uv_version": "0.11.29",
        "python_version": "3.12.4",
        "uhem_python_module": "Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0",
        "sync_mode": "uv_sync_frozen_extra_gpu",
        "sync_command": "uv sync --frozen --extra gpu --python $(command -v python3)",
        "pyproject_sha256": "c91fdd03ae9705565572eee31924d4c0bca24bf5431a8eabff4c061882f94929",
        "uv_lock_sha256": "de7891b832854162111208644ddb72685069ce8128e2bed9dbb7993aa6af5861",
        "relative_exclude_newer_lock_requires_frozen": True,
    }
    if not isinstance(code, Mapping) or code.get("training_environment") != expected_environment:
        raise StrictTrainingError("family recipe training environment drifted")
    training = recipe.get("training")
    if not isinstance(training, Mapping):
        raise StrictTrainingError("family recipe has no training object")
    required_training = {
        "optimizer": "muon_adamw",
        "precision": "bfloat16",
        "fp8_enabled": False,
        "gradient_clip_norm": 0.0,
        "lr_schedule": "wsd",
        "cooldown_fraction": 0.1,
        "cooldown_final_lr_fraction": 0.0,
        "data_order": "bestfit",
        "target_param_count": "scaling",
        "target_param_data_ratio": -1.0,
        "strict_transactional_checkpoints": True,
        "fixed_validation": True,
    }
    for field, expected in required_training.items():
        if training.get(field) != expected:
            raise StrictTrainingError(f"family recipe training.{field} drifted")
    model = recipe.get("model")
    if not isinstance(model, Mapping):
        raise StrictTrainingError("family recipe has no model object")
    required_model = {
        "depth": 32,
        "aspect_ratio": 64,
        "model_dim": 2048,
        "head_dim": 128,
        "num_heads": 16,
        "num_kv_heads": 16,
        "max_seq_len": 2048,
        "vocab_size": 32768,
        "parameter_ratio_convention": "nanochat_scaling",
    }
    for field, expected in required_model.items():
        if model.get(field) != expected:
            raise StrictTrainingError(f"family recipe model.{field} drifted")
    gate = recipe.get("attention_backend_gate")
    if not isinstance(gate, Mapping):
        raise StrictTrainingError("family recipe has no attention backend gate")
    if (
        (gate.get("preferred_backend"), gate.get("preferred_window_pattern"))
        != ("fa3", "SSSL")
        or (gate.get("fallback_backend"), gate.get("fallback_window_pattern"))
        != ("sdpa", "L")
        or gate.get("required_gpu_family") != "A100"
    ):
        raise StrictTrainingError("attention backend gate policy drifted")
    evaluation = training.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("fixed_validation_full_manifest") is not True
        or evaluation.get("eval_tokens_cli_unused") != -1
    ):
        raise StrictTrainingError("strict evaluation must exhaust the frozen manifest")


def load_artifact_bindings(
    *,
    study_manifest_path: str | Path,
    data_dir: str | Path,
    exposure_plan_path: str | Path,
    validation_manifest_path: str | Path,
    tokenizer_sha256: str,
    seed: int,
    world_size: int,
    num_iterations: int,
    total_batch_size: int,
) -> ArtifactBindings:
    """Load and cross-bind all non-tokenizer training artifacts."""

    recipe = load_json_strict(study_manifest_path)
    if not isinstance(recipe, dict):
        raise StrictTrainingError("study manifest must contain an object")
    validate_family_recipe(recipe)
    recipe_sha = verify_manifest_hash(recipe)
    dataset = load_json_strict(Path(data_dir) / "fineweb2_manifest.json")
    if not isinstance(dataset, dict):
        raise StrictTrainingError("dataset manifest must contain an object")
    validate_dataset_manifest(dataset, profile="strict")
    dataset_sha = verify_manifest_hash(dataset)
    corpus_id = recipe["artifacts"]["corpus_id"]
    if (
        dataset.get("dataset", {}).get("repo_id") != f"local-composite/{corpus_id}"
        or dataset.get("metadata", {}).get("corpus_name") != corpus_id
    ):
        raise StrictTrainingError("dataset manifest uses another family corpus")
    exposure = load_json_strict(exposure_plan_path)
    if not isinstance(exposure, dict):
        raise StrictTrainingError("training exposure plan must contain an object")
    validate_training_exposure_plan(exposure)
    exposure_sha = verify_manifest_hash(exposure)
    expected_plan = {
        "estimand": "equal_token",
        "source_dataset_manifest_sha256": dataset_sha,
        "study_sha256": recipe_sha,
        "tokenizer_sha256": tokenizer_sha256,
        "seed": seed,
        "world_size": world_size,
        "data_order": "bestfit",
        "horizon": {
            "unit": "token_positions",
            "value": num_iterations * total_batch_size,
        },
    }
    for field, expected in expected_plan.items():
        if exposure.get(field) != expected:
            raise StrictTrainingError(f"training exposure binding mismatch at {field}")
    if exposure.get("derived", {}).get("optimizer_steps") not in {
        None,
        num_iterations,
    }:
        raise StrictTrainingError("training exposure optimizer-step count drifted")
    validation = load_json_strict(validation_manifest_path)
    if not isinstance(validation, dict):
        raise StrictTrainingError("validation manifest must contain an object")
    validate_exposure_manifest(validation, source_dataset_manifest=dataset)
    validation_sha = verify_manifest_hash(validation)
    if validation.get("mode") != "validation":
        raise StrictTrainingError("validation manifest is not validation mode")
    if validation.get("study_sha256") not in {None, recipe_sha}:
        raise StrictTrainingError("validation manifest is bound to another study")
    return ArtifactBindings(
        recipe=recipe,
        recipe_sha256=recipe_sha,
        dataset_manifest=dataset,
        dataset_sha256=dataset_sha,
        exposure_plan=exposure,
        exposure_plan_sha256=exposure_sha,
        validation_manifest=validation,
        validation_sha256=validation_sha,
    )


def validate_preflight_artifact_bindings(
    preflight_receipt_path: str | Path,
    *,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    code_revision: str,
    data_dir: str | Path,
    tokenizer_sha256: str,
    dataset_manifest: Mapping[str, Any],
    dataset_sha256: str,
    validation_manifest: Mapping[str, Any],
    validation_sha256: str,
    exposure_plan_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    """Bind every runtime data artifact to the sealed family preflight.

    This check deliberately runs before CUDA/model construction.  It prevents
    proxy, smoke, signal-smoke, and production jobs alike from spending GPU
    time with a fresh tokenizer, corpus, validation exposure, or training plan
    paired with a stale preflight receipt.
    """

    preflight = load_json_strict(preflight_receipt_path)
    if not isinstance(preflight, dict):
        raise StrictTrainingError("preflight receipt must contain an object")
    try:
        preflight_sha = verify_manifest_hash(preflight)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid sealed preflight receipt: {exc}") from exc
    expected_preflight = {
        "kind": "d32_family_preflight_receipt",
        "family_id": recipe["family_id"],
    }
    for field, wanted in expected_preflight.items():
        if preflight.get(field) != wanted:
            raise StrictTrainingError(f"preflight artifact binding mismatch at {field}")
    recipe_record = preflight.get("recipe")
    code_record = preflight.get("code")
    tokenizer_record = preflight.get("tokenizer")
    corpus_record = preflight.get("corpus")
    mixture_record = preflight.get("mixture_config")
    if not all(
        isinstance(record, Mapping)
        for record in (
            recipe_record,
            code_record,
            tokenizer_record,
            corpus_record,
            mixture_record,
        )
    ):
        raise StrictTrainingError("preflight artifact inventory is malformed")
    if recipe_record.get("canonical_sha256") != recipe_sha256:
        raise StrictTrainingError("preflight uses another family recipe")
    if code_record.get("git_commit") != code_revision:
        raise StrictTrainingError("preflight uses another code revision")
    environment = recipe.get("code_provenance", {}).get("training_environment")
    if not isinstance(environment, Mapping):
        raise StrictTrainingError("family recipe lacks its training environment")
    expected_environment_record = {
        "pyproject_sha256": environment["pyproject_sha256"],
        "uv_lock_sha256": environment["uv_lock_sha256"],
        "uv_version": environment["uv_version"],
        "python_version": environment["python_version"],
        "environment_sync_mode": environment["sync_mode"],
    }
    for field, wanted in expected_environment_record.items():
        if code_record.get(field) != wanted:
            raise StrictTrainingError(
                f"preflight training environment mismatch at {field}"
            )
    if tokenizer_record.get("package_manifest_sha256") != tokenizer_sha256:
        raise StrictTrainingError("preflight uses another tokenizer package")

    artifacts = family_artifact_contract(str(recipe["family_id"]))
    if dict(recipe.get("artifacts", {})) != artifacts:
        raise StrictTrainingError("runtime recipe artifact contract drifted")
    corpus_id = artifacts["corpus_id"]
    mixture_path = Path(str(mixture_record.get("path", ""))).resolve()
    if not mixture_path.is_file() or mixture_path.is_symlink():
        raise StrictTrainingError("preflight mixture config is missing or unsafe")
    mixture = load_json_strict(mixture_path)
    if not isinstance(mixture, dict):
        raise StrictTrainingError("preflight mixture config must contain an object")
    policy_sha = hashlib.sha256(canonical_json(mixture).encode("utf-8")).hexdigest()
    if (
        mixture.get("name") != corpus_id
        or mixture_record.get("sha256") != file_sha256(mixture_path)
        or mixture_record.get("policy_sha256") != policy_sha
        or mixture_record.get("corpus_name") != corpus_id
    ):
        raise StrictTrainingError("preflight mixture/policy family binding drifted")

    root = Path(data_dir).resolve()
    if Path(str(corpus_record.get("root", ""))).resolve() != root:
        raise StrictTrainingError("preflight uses another corpus root")
    if corpus_record.get("name") != corpus_id:
        raise StrictTrainingError("preflight uses another corpus name")
    if corpus_record.get("dataset_manifest_sha256") != dataset_sha256:
        raise StrictTrainingError("preflight uses another dataset manifest")
    if corpus_record.get("validation_exposure_manifest_sha256") != validation_sha256:
        raise StrictTrainingError("preflight uses another validation exposure")

    selection = validation_manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise StrictTrainingError("validation exposure lacks selection accounting")
    if (
        corpus_record.get("validation_payload_bytes")
        != selection.get("realized_payload_bytes")
        or corpus_record.get("validation_documents")
        != selection.get("realized_documents")
    ):
        raise StrictTrainingError("preflight validation coverage accounting drifted")

    corpus_manifest = load_json_strict(root / "corpus_manifest.json")
    if not isinstance(corpus_manifest, dict):
        raise StrictTrainingError("final corpus manifest must contain an object")
    try:
        corpus_sha = verify_manifest_hash(corpus_manifest)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid final corpus manifest: {exc}") from exc
    if (
        corpus_record.get("manifest_sha256") != corpus_sha
        or corpus_manifest.get("schema_version") != "1.0"
        or corpus_manifest.get("kind") != "turkish_pretrain_corpus"
        or corpus_manifest.get("name") != corpus_id
        or corpus_manifest.get("policy_sha256") != policy_sha
        or corpus_manifest.get("source_receipt_sha256")
        != corpus_record.get("source_receipt_sha256")
        or corpus_manifest.get("nanochat_dataset_manifest_sha256") != dataset_sha256
        or corpus_manifest.get("tokenizer", {}).get("name")
        != artifacts["tokenizer_name"]
        or corpus_manifest.get("tokenizer", {}).get("package_sha256")
        != tokenizer_sha256
    ):
        raise StrictTrainingError("preflight final-corpus binding drifted")
    if (
        dataset_manifest.get("dataset", {}).get("repo_id")
        != f"local-composite/{corpus_id}"
        or dataset_manifest.get("metadata", {}).get("corpus_name") != corpus_id
    ):
        raise StrictTrainingError("runtime dataset/corpus family binding drifted")

    source_path = (root / artifacts["source_receipt"]).resolve()
    if not source_path.is_file() or source_path.is_symlink():
        raise StrictTrainingError("source receipt is missing or unsafe")
    source_receipt = load_json_strict(source_path)
    if not isinstance(source_receipt, dict):
        raise StrictTrainingError("source receipt must contain an object")
    try:
        source_sha = verify_manifest_hash(source_receipt)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid source receipt: {exc}") from exc
    if (
        source_receipt.get("schema_version") != "1.0"
        or source_receipt.get("kind") != "turkish_pretrain_source_receipt"
        or source_sha != corpus_record.get("source_receipt_sha256")
        or source_receipt.get("policy_sha256") != policy_sha
    ):
        raise StrictTrainingError("runtime source-receipt policy binding drifted")
    if recipe["family_id"] in _REPEAT_CAPACITY_FAMILY_IDS:
        try:
            from nanochat.turkish_corpus import validate_source_receipt

            validate_source_receipt(source_receipt, mixture)
        except (OSError, ValueError) as exc:
            raise StrictTrainingError(
                f"runtime source receipt failed full policy validation: {exc}"
            ) from exc

    derived_sources = source_receipt.get("derived_sources")
    if not isinstance(derived_sources, Mapping):
        raise StrictTrainingError("source receipt derived-source inventory is malformed")
    macocu_relative = artifacts.get("macocu_preparation_manifest")
    macocu_record = corpus_record.get("macocu_preparation_manifest")
    if macocu_relative is None:
        if derived_sources or macocu_record is not None:
            raise StrictTrainingError("v1 family unexpectedly binds derived MaCoCu data")
    else:
        if not isinstance(macocu_record, Mapping):
            raise StrictTrainingError(
                "derived-data family preflight lacks its MaCoCu preparation manifest"
            )
        base_dir = Path(str(preflight.get("base_dir", ""))).resolve()
        expected_macocu_path = (base_dir / macocu_relative).resolve()
        actual_macocu_path = Path(str(macocu_record.get("path", ""))).resolve()
        if actual_macocu_path != expected_macocu_path:
            raise StrictTrainingError("preflight MaCoCu preparation path drifted")
        if not actual_macocu_path.is_file() or actual_macocu_path.is_symlink():
            raise StrictTrainingError("MaCoCu preparation manifest is missing or unsafe")
        macocu_manifest = load_json_strict(actual_macocu_path)
        if not isinstance(macocu_manifest, dict):
            raise StrictTrainingError("MaCoCu preparation manifest must be an object")
        try:
            macocu_sha = verify_manifest_hash(macocu_manifest)
        except ValueError as exc:
            raise StrictTrainingError(
                f"invalid MaCoCu preparation manifest: {exc}"
            ) from exc
        source_macocu = derived_sources.get("macocu_genre_tr")
        if (
            macocu_manifest.get("schema_version") != "1.0"
            or macocu_manifest.get("kind") != "turkish_macocu_genre_preparation"
            or not isinstance(source_macocu, Mapping)
            or macocu_record.get("sha256") != macocu_sha
            or source_macocu.get("manifest_sha256") != macocu_sha
            or source_macocu.get("manifest_uri") != actual_macocu_path.as_uri()
        ):
            raise StrictTrainingError("MaCoCu preparation/source receipt binding drifted")

    base_dir = Path(str(preflight.get("base_dir", ""))).resolve()
    for source_id, artifact_key in (
        ("mot_tr_v1_11", "mot_preparation_manifest"),
        ("parlamint_tr_v5_0", "parlamint_preparation_manifest"),
    ):
        relative = artifacts.get(artifact_key)
        preflight_record = corpus_record.get(artifact_key)
        if relative is None:
            if source_id in derived_sources or preflight_record is not None:
                raise StrictTrainingError(
                    f"family unexpectedly binds {source_id} prepared data"
                )
            continue
        _validate_anchor_preflight_binding(
            (base_dir / str(relative)).resolve(),
            source_id=source_id,
            derived_sources=derived_sources,
            preflight_record=preflight_record,
        )

    capacity_path = (root / "packing_capacity_receipt.json").resolve()
    capacity = load_json_strict(capacity_path)
    if not isinstance(capacity, dict):
        raise StrictTrainingError("packing capacity receipt must contain an object")
    try:
        capacity_sha = verify_manifest_hash(capacity)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid packing capacity receipt: {exc}") from exc
    preflight_capacity = corpus_record.get("packing_capacity_receipt")
    if not isinstance(preflight_capacity, Mapping):
        raise StrictTrainingError("preflight lacks packing-capacity evidence")
    if (
        Path(str(preflight_capacity.get("path", ""))).resolve() != capacity_path
        or preflight_capacity.get("sha256") != capacity_sha
        or preflight_capacity.get("gate_passed") is not True
        or capacity.get("gate_passed") is not True
        or corpus_manifest.get("packing_capacity", {}).get("sha256") != capacity_sha
    ):
        raise StrictTrainingError("preflight packing-capacity binding drifted")

    exposure_index = load_json_strict(root / "exposure_plan_index.json")
    if not isinstance(exposure_index, dict):
        raise StrictTrainingError("exposure plan index must contain an object")
    try:
        exposure_index_sha = verify_manifest_hash(exposure_index)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid exposure plan index: {exc}") from exc
    current_plan_in_index = any(
        isinstance(record, Mapping) and record.get("sha256") == exposure_plan_sha256
        for record in exposure_index.get("plans", [])
    )
    if (
        corpus_record.get("exposure_plan_index_sha256") != exposure_index_sha
        or exposure_index.get("schema_version") != "1.0"
        or exposure_index.get("family_id") != recipe["family_id"]
        or exposure_index.get("study_manifest_sha256") != recipe_sha256
        or exposure_index.get("source_dataset_manifest_sha256") != dataset_sha256
        or exposure_index.get("tokenizer_artifact_sha256") != tokenizer_sha256
        or exposure_index.get("packing_capacity_receipt_sha256") != capacity_sha
        or exposure_index.get("validation", {}).get("sha256") != validation_sha256
        or not current_plan_in_index
    ):
        raise StrictTrainingError("preflight exposure-plan index binding drifted")
    preflight_plans = corpus_record.get("training_exposure_plans")
    if not isinstance(preflight_plans, Mapping) or not any(
        isinstance(record, Mapping) and record.get("sha256") == exposure_plan_sha256
        for record in preflight_plans.values()
    ):
        raise StrictTrainingError("preflight does not inventory the current exposure plan")

    validation_relative = corpus_record.get("validation_file")
    if not isinstance(validation_relative, str) or not validation_relative:
        raise StrictTrainingError("preflight validation file path is invalid")
    validation_path = (root / validation_relative).resolve()
    try:
        validation_path.relative_to(root)
    except ValueError as exc:
        raise StrictTrainingError("preflight validation file escapes the corpus") from exc
    if (
        dataset_manifest.get("validation_file") != validation_relative
        or not validation_path.is_file()
        or validation_path.is_symlink()
        or validation_path.stat().st_size
        != corpus_record.get("validation_file_size_bytes")
        or file_sha256(validation_path) != corpus_record.get("validation_file_sha256")
    ):
        raise StrictTrainingError("preflight validation file binding drifted")
    return preflight, preflight_sha, capacity_sha


def validate_attention_probe_receipt(
    attention_probe_path: str | Path,
    preflight_receipt_path: str | Path,
    *,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    code_revision: str,
    attention_backend: str,
    window_pattern: str,
) -> tuple[dict[str, Any], str]:
    """Verify the sealed A100 probe that selects the runtime attention pair."""

    preflight = load_json_strict(preflight_receipt_path)
    if not isinstance(preflight, dict):
        raise StrictTrainingError("preflight receipt must contain an object")
    try:
        preflight_sha = verify_manifest_hash(preflight)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid sealed preflight receipt: {exc}") from exc
    if preflight.get("kind") != "d32_family_preflight_receipt":
        raise StrictTrainingError("attention probe preflight receipt kind mismatch")
    if preflight.get("recipe", {}).get("canonical_sha256") != recipe_sha256:
        raise StrictTrainingError("attention preflight receipt uses another recipe")
    if preflight.get("code", {}).get("git_commit") != code_revision:
        raise StrictTrainingError("attention preflight receipt uses another code revision")

    probe = load_json_strict(attention_probe_path)
    if not isinstance(probe, dict):
        raise StrictTrainingError("attention probe receipt must contain an object")
    try:
        probe_sha = verify_manifest_hash(probe)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid sealed attention probe: {exc}") from exc
    expected = {
        "kind": "d32_attention_backend_probe",
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha256,
        "preflight_receipt_sha256": preflight_sha,
        "code_revision": code_revision,
        "world_size": 1,
    }
    for field, wanted in expected.items():
        if probe.get(field) != wanted:
            raise StrictTrainingError(f"attention probe mismatch at {field}")
    gate = recipe["attention_backend_gate"]
    gpu = probe.get("gpu")
    if not isinstance(gpu, Mapping) or str(gpu.get("name", "")).upper().find(
        str(gate["required_gpu_family"]).upper()
    ) < 0:
        raise StrictTrainingError("attention probe did not run on the required GPU family")
    detection = probe.get("module_detection")
    if not isinstance(detection, Mapping):
        raise StrictTrainingError("attention probe lacks module detection")
    selected = (
        detection.get("selected_backend_after_probe"),
        detection.get("selected_window_pattern"),
    )
    preferred = (gate["preferred_backend"], gate["preferred_window_pattern"])
    fallback = (gate["fallback_backend"], gate["fallback_window_pattern"])
    if selected not in {preferred, fallback}:
        raise StrictTrainingError("attention probe selected a pair outside recipe policy")
    if selected != (attention_backend, window_pattern):
        raise StrictTrainingError("runtime backend/window differs from attention probe")
    expected_flash_hash = recipe["code_provenance"]["exact_file_sha256"][
        "nanochat/flash_attention.py"
    ]
    if detection.get("flash_attention_file_sha256") != expected_flash_hash:
        raise StrictTrainingError("attention probe used an unreviewed attention runtime")
    reason = probe.get("selection_reason")
    if not isinstance(reason, str) or not reason:
        raise StrictTrainingError("attention probe lacks a selection reason")
    if selected == preferred:
        if (
            probe.get("decision") != "accepted_fa3_SSSL"
            or detection.get("HAS_FA3_at_import") is not True
            or detection.get("USE_FA3_at_import") is not True
        ):
            raise StrictTrainingError("preferred FA3+SSSL pair was not accepted safely")
    elif probe.get("decision") != "accepted_sdpa_L_fallback":
        raise StrictTrainingError("SDPA+L fallback receipt is contradictory")
    full_model = probe.get("selected_d32_model_forward_backward")
    expected_config = {
        "depth": 32,
        "model_dim": 2048,
        "num_heads": 16,
        "num_kv_heads": 16,
        "head_dim": 128,
        "max_seq_len": 2048,
        "vocab_size": 32768,
    }
    if not isinstance(full_model, Mapping):
        raise StrictTrainingError("attention probe lacks the selected d32 model smoke")
    expected_full_model = {
        "backend": selected[0],
        "window_pattern": selected[1],
        "config": expected_config,
        "construction": "meta_then_to_empty_cuda_then_literal_seed42_init_weights",
        "initialization_seed": 42,
        "batch_sequences": 1,
        "sequence_length": 2048,
        "compute_dtype": "torch.bfloat16",
        "output_finite": True,
        "loss_finite": True,
        "gradients_present": True,
        "gradients_finite": True,
    }
    for field, wanted in expected_full_model.items():
        if full_model.get(field) != wanted:
            raise StrictTrainingError(f"selected d32 attention smoke mismatch at {field}")
    expected_parameter_dtypes = {
        "torch.bfloat16": {
            "tensor_count": 17,
            "element_count": 1_140_850_688,
        },
        "torch.float32": {
            "tensor_count": 214,
            "element_count": 1_677_724_762,
        },
    }
    parameter_dtypes = full_model.get("parameter_dtype_inventory")
    if parameter_dtypes != expected_parameter_dtypes:
        raise StrictTrainingError("selected d32 attention smoke parameter dtypes drifted")
    if sum(
        int(record["element_count"])
        for record in parameter_dtypes.values()
    ) != 2_818_575_450:
        raise StrictTrainingError("selected d32 attention smoke parameter count drifted")
    if full_model.get("buffer_dtype_inventory") != {
        "torch.float32": {"tensor_count": 2, "element_count": 2_621_440}
    }:
        raise StrictTrainingError("selected d32 attention smoke buffer dtypes drifted")
    if (
        not isinstance(full_model.get("gradient_tensor_count"), int)
        or full_model["gradient_tensor_count"] <= 0
        or not isinstance(full_model.get("peak_cuda_memory_bytes"), int)
        or full_model["peak_cuda_memory_bytes"] <= 0
    ):
        raise StrictTrainingError("selected d32 attention smoke lacks gradient/memory evidence")
    benchmarks = probe.get("pattern_benchmarks")
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != {"L", "SSSL"}:
        raise StrictTrainingError("attention probe lacks both pattern benchmarks")
    return probe, probe_sha


def capacity_world_gate_record(
    receipt: Mapping[str, Any], world_size: int
) -> dict[str, Any]:
    """Return compact, protocol-safe evidence for one capacity topology."""

    simulation = receipt.get("simulation")
    if not isinstance(simulation, Mapping):
        raise StrictTrainingError("capacity receipt lacks simulation evidence")
    worlds = simulation.get("worlds")
    if not isinstance(worlds, Mapping):
        raise StrictTrainingError("capacity receipt lacks topology evidence")
    world = worlds.get(str(world_size))
    if not isinstance(world, Mapping):
        raise StrictTrainingError(
            f"capacity receipt lacks ws{world_size} evidence"
        )
    if receipt.get("kind") != "turkish_bestfit_repeat_capacity_receipt":
        return dict(world)
    horizons = world.get("horizons")
    if not isinstance(horizons, Mapping):
        raise StrictTrainingError("repetition capacity lacks horizon evidence")
    margin = horizons.get("s40_margin")
    if not isinstance(margin, Mapping):
        raise StrictTrainingError("repetition capacity lacks the s40 margin")
    authorized = margin.get("scheduled_token_positions")
    if isinstance(authorized, bool) or not isinstance(authorized, int) or authorized <= 0:
        raise StrictTrainingError("repetition capacity has an invalid authorized horizon")
    return {
        "capacity_mode": "whole_pool_repeat_v3",
        "world_size": world_size,
        "device_batch_sequences": world.get("device_batch_sequences"),
        "max_seq_len": world.get("max_seq_len"),
        "gradient_accumulation_steps": world.get("gradient_accumulation_steps"),
        "authorized_global_scheduled_positions": authorized,
        "required_positions_with_margin": authorized,
        "repetition_tier": world.get("repetition_tier"),
        "max_loaded_epoch": margin.get("max_loaded_epoch"),
        "max_consumed_epoch": margin.get("max_consumed_epoch"),
        "epoch5_loaded_including_prefetch": margin.get(
            "epoch5_loaded_including_prefetch"
        ),
        "whole_pool_repetition_only": world.get("whole_pool_repetition_only"),
        "source_specific_repetition": world.get("source_specific_repetition"),
    }


def capacity_authorized_positions(selected_capacity: Mapping[str, Any]) -> int:
    """Read the generic authorized horizon while preserving v1/v2 receipts."""

    value = selected_capacity.get("authorized_global_scheduled_positions")
    if value is None:
        value = selected_capacity.get("safe_global_scheduled_positions")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StrictTrainingError("selected capacity has no valid authorized horizon")
    return value


def validate_bestfit_capacity_receipt(
    receipt_path: str | Path,
    *,
    data_dir: str | Path,
    dataset_manifest: Mapping[str, Any],
    dataset_sha256: str,
    tokenizer_sha256: str,
    family_id: str,
    recipe_sha256: str,
    exposure_plan_sha256: str,
    world_size: int,
    required_token_positions: int,
    batch_sequences: int,
    sequence_length: int,
    global_batch_tokens: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Verify the family-specific exact capacity gate for one topology."""

    family_artifact_contract(family_id)
    root = Path(data_dir).resolve()
    resolved_receipt_path = Path(receipt_path).resolve()
    expected_receipt_path = (root / "packing_capacity_receipt.json").resolve()
    if resolved_receipt_path != expected_receipt_path:
        raise StrictTrainingError(
            "--packing-capacity-receipt must name data_dir/packing_capacity_receipt.json"
        )
    receipt = load_json_strict(resolved_receipt_path)
    if not isinstance(receipt, dict):
        raise StrictTrainingError("packing capacity receipt must contain an object")
    try:
        receipt_sha = verify_manifest_hash(receipt)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid packing capacity receipt: {exc}") from exc
    if family_id in _REPEAT_CAPACITY_FAMILY_IDS:
        try:
            from nanochat.packing_capacity import (
                validate_repetition_capacity_receipt,
            )

            summary = validate_repetition_capacity_receipt(
                receipt,
                dataset_manifest_sha256=dataset_sha256,
                tokenizer_package_sha256=tokenizer_sha256,
                # Manual-risk production remains fail-closed until the exact
                # separately sealed approval is added to the family artifact
                # contract and preflight inventory.
                manual_repetition_risk_approval=None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StrictTrainingError(
                f"invalid repetition capacity receipt: {exc}"
            ) from exc
        if (
            summary.get("canonical_sha256") != receipt_sha
            or summary.get("gate_passed") is not True
            or summary.get("cleanup_authorized") is not True
            or summary.get("approval_satisfied") is not True
        ):
            raise StrictTrainingError("repetition capacity gate did not pass")
        final_manifest = load_json_strict(root / "corpus_manifest.json")
        if not isinstance(final_manifest, dict):
            raise StrictTrainingError("final corpus manifest must contain an object")
        try:
            verify_manifest_hash(final_manifest)
        except ValueError as exc:
            raise StrictTrainingError(
                f"invalid final corpus manifest: {exc}"
            ) from exc
        if (
            final_manifest.get("nanochat_dataset_manifest_sha256") != dataset_sha256
            or final_manifest.get("tokenizer", {}).get("package_sha256")
            != tokenizer_sha256
            or final_manifest.get("packing_capacity", {}).get("sha256")
            != receipt_sha
            or final_manifest.get("packing_capacity", {}).get("all_worlds_pass")
            is not True
            or final_manifest.get("packing_capacity", {}).get("cleanup_authorized")
            is not True
        ):
            raise StrictTrainingError(
                "final corpus does not bind the passing repetition capacity receipt"
            )
        validation_policy = dataset_manifest.get("metadata", {}).get(
            "validation_policy"
        )
        final_validation_policy = final_manifest.get(
            "validation_whole_document_no_crop"
        )
        for label, policy in (
            ("dataset", validation_policy),
            ("final corpus", final_validation_policy),
        ):
            if (
                not isinstance(policy, Mapping)
                or policy.get("policy") != "whole_document_no_crop"
                or policy.get("max_payload_tokens") != sequence_length
                or policy.get("max_encoded_tokens_with_bos") != sequence_length + 1
                or policy.get("oversized_document_action")
                != "excluded_before_exposure_selection"
            ):
                raise StrictTrainingError(
                    f"{label} validation no-crop policy mismatch"
                )
        selected = capacity_world_gate_record(receipt, world_size)
        if (
            selected.get("device_batch_sequences") != batch_sequences
            or selected.get("max_seq_len") != sequence_length
            or selected.get("authorized_global_scheduled_positions", 0)
            < required_token_positions
        ):
            raise StrictTrainingError(
                "selected repetition capacity does not cover this runtime horizon"
            )
        index = load_json_strict(root / "exposure_plan_index.json")
        if not isinstance(index, dict):
            raise StrictTrainingError("exposure plan index must contain an object")
        try:
            verify_manifest_hash(index)
        except ValueError as exc:
            raise StrictTrainingError(
                f"invalid exposure plan index: {exc}"
            ) from exc
        if (
            index.get("schema_version") != "1.0"
            or index.get("kind") != "d32_exposure_plan_index"
            or index.get("family_id") != family_id
            or index.get("study_manifest_sha256") != recipe_sha256
            or index.get("source_dataset_manifest_sha256") != dataset_sha256
            or index.get("tokenizer_artifact_sha256") != tokenizer_sha256
            or index.get("packing_capacity_receipt_sha256") != receipt_sha
            or not any(
                isinstance(record, Mapping)
                and record.get("sha256") == exposure_plan_sha256
                for record in index.get("plans", [])
            )
        ):
            raise StrictTrainingError(
                "exposure plan index does not bind capacity/current plan"
            )
        return receipt, receipt_sha, selected
    expected_receipt = {
        "kind": "turkish_bestfit_capacity_receipt",
        "dataset_manifest_sha256": dataset_sha256,
        "tokenizer_package_sha256": tokenizer_sha256,
        "mix_gate_evaluated_on_common_horizon": True,
        "mix_gate_passed": True,
        "no_wrap_gate_passed": True,
        "gate_passed": True,
        "cleanup_authorized": True,
        "recommendation_requires_fresh_simulation": True,
    }
    for field, wanted in expected_receipt.items():
        if receipt.get(field) != wanted:
            raise StrictTrainingError(f"packing capacity receipt mismatch at {field}")
    final_manifest = load_json_strict(root / "corpus_manifest.json")
    if not isinstance(final_manifest, dict):
        raise StrictTrainingError("final corpus manifest must contain an object")
    try:
        verify_manifest_hash(final_manifest)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid final corpus manifest: {exc}") from exc
    if (
        final_manifest.get("nanochat_dataset_manifest_sha256") != dataset_sha256
        or final_manifest.get("tokenizer", {}).get("package_sha256") != tokenizer_sha256
        or final_manifest.get("packing_capacity", {}).get("sha256") != receipt_sha
        or final_manifest.get("packing_capacity", {}).get("all_worlds_pass") is not True
    ):
        raise StrictTrainingError("final corpus does not bind the passing capacity receipt")
    validation_policy = dataset_manifest.get("metadata", {}).get("validation_policy")
    final_validation_policy = final_manifest.get("validation_whole_document_no_crop")
    for label, policy in (
        ("dataset", validation_policy),
        ("final corpus", final_validation_policy),
    ):
        if (
            not isinstance(policy, Mapping)
            or policy.get("policy") != "whole_document_no_crop"
            or policy.get("max_payload_tokens") != sequence_length
            or policy.get("max_encoded_tokens_with_bos") != sequence_length + 1
            or policy.get("oversized_document_action")
            != "excluded_before_exposure_selection"
        ):
            raise StrictTrainingError(f"{label} validation no-crop policy mismatch")
    simulation = receipt.get("simulation")
    if not isinstance(simulation, Mapping):
        raise StrictTrainingError("capacity receipt lacks simulation")
    expected_simulation = {
        "implementation": "nanochat_upstream_bos_bestfit_crop_capacity_v2",
        "all_worlds_pass": True,
    }
    for field, wanted in expected_simulation.items():
        if simulation.get(field) != wanted:
            raise StrictTrainingError(f"capacity simulation mismatch at {field}")
    implementation_path = Path(__file__).with_name("packing_capacity.py")
    if simulation.get("implementation_file_sha256") != file_sha256(
        implementation_path
    ):
        raise StrictTrainingError("capacity simulator source differs from its receipt")
    upstream = simulation.get("upstream_contract")
    if not isinstance(upstream, Mapping) or upstream != {
        "nanochat_revision": PINNED_UPSTREAM_REVISION[:8],
        "encode_call": "tokenizer.encode(doc_batch, prepend=bos_token, num_threads=4)",
        "tokenizer_batch_size": 128,
        "tokenizer_threads": 4,
        "refill_buffer_size": 1000,
        "tie_breaks": "first_largest_fit_else_first_shortest",
        "cropped_tail_policy": "discard",
        "rank_sharding": "row_group_index_mod_world_size",
    }:
        raise StrictTrainingError("capacity simulation upstream contract drifted")
    parity = simulation.get("fixture_parity")
    if (
        not isinstance(parity, Mapping)
        or parity.get("passed") is not True
        or parity.get("upstream_commit") != PINNED_UPSTREAM_REVISION
    ):
        raise StrictTrainingError("capacity simulator lacks pinned-upstream parity")
    worlds = simulation.get("worlds")
    if not isinstance(worlds, Mapping) or set(worlds) != {"8", "16"}:
        raise StrictTrainingError("capacity receipt must prove both ws8 and ws16")
    selected = worlds.get(str(world_size))
    if not isinstance(selected, Mapping):
        raise StrictTrainingError("capacity receipt lacks the selected topology")
    selected_expected = {
        "world_size": world_size,
        "device_batch_sequences": batch_sequences,
        "max_seq_len": sequence_length,
        "buffer_size": 1000,
        "preserve_document_tails": False,
        "row_capacity": sequence_length + 1,
        "rank_sharding": "parquet_row_group_index_mod_world_size",
        "passes_40x_no_wrap_with_margin": True,
        "first_wrap_observation": "right_censored_at_required_horizon",
        "safe_global_scheduled_positions_semantics": (
            "right_censored_proven_lower_bound_at_required_horizon"
        ),
        "aggregate_scope": "exact_common_required_horizon_all_ranks",
    }
    for field, wanted in selected_expected.items():
        if selected.get(field) != wanted:
            raise StrictTrainingError(f"selected capacity topology mismatch at {field}")
    if (
        selected.get("required_optimizer_steps") != 32_000
        or selected.get("safety_margin_fraction") != 0.02
        or selected.get("required_optimizer_steps_with_margin") != 32_640
        or selected.get("required_positions_with_margin", 0)
        != 32_640 * global_batch_tokens
        or selected.get("safe_global_scheduled_positions", 0)
        < required_token_positions
        or selected.get("common_prefix_scheduled_positions", 0)
        < required_token_positions
        or len(selected.get("completed_microbatches_by_rank", [])) != world_size
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < selected.get("requested_microbatches_per_rank", 0)
            for value in selected.get("completed_microbatches_by_rank", [])
        )
        or len(selected.get("first_wrap_before_microbatch_by_rank", []))
        != world_size
        or any(
            value is not None
            for value in selected.get("first_wrap_before_microbatch_by_rank", [])
        )
    ):
        raise StrictTrainingError("selected capacity receipt does not cover this horizon")
    index = load_json_strict(root / "exposure_plan_index.json")
    if not isinstance(index, dict):
        raise StrictTrainingError("exposure plan index must contain an object")
    try:
        verify_manifest_hash(index)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid exposure plan index: {exc}") from exc
    if (
        index.get("schema_version") != "1.0"
        or index.get("kind") != "d32_exposure_plan_index"
        or index.get("family_id") != family_id
        or index.get("study_manifest_sha256") != recipe_sha256
        or index.get("source_dataset_manifest_sha256") != dataset_sha256
        or index.get("tokenizer_artifact_sha256") != tokenizer_sha256
        or index.get("packing_capacity_receipt_sha256") != receipt_sha
        or not any(
            isinstance(record, Mapping)
            and record.get("sha256") == exposure_plan_sha256
            for record in index.get("plans", [])
        )
    ):
        raise StrictTrainingError("exposure plan index does not bind capacity/current plan")
    return receipt, receipt_sha, dict(selected)


def verify_live_fa3_kernel_inventory(
    probe: Mapping[str, Any], live_fa3_module: Any
) -> str | None:
    """Require production FA3 bytes to equal the sealed probe inventory."""

    detection = probe.get("module_detection")
    if not isinstance(detection, Mapping):
        raise StrictTrainingError("attention probe lacks module detection")
    if detection.get("selected_backend_after_probe") != "fa3":
        if live_fa3_module is not None and detection.get("selected_backend_after_probe") != "sdpa":
            raise StrictTrainingError("attention probe backend is invalid")
        return None
    expected = detection.get("fa3_kernel_inventory")
    if not isinstance(expected, Mapping):
        raise StrictTrainingError("FA3 probe lacks a sealed kernel inventory")
    try:
        module_file = Path(inspect.getfile(live_fa3_module)).resolve()
    except (OSError, TypeError) as exc:
        raise StrictTrainingError("cannot locate the live FA3 module") from exc
    root = module_file.parent
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if len(paths) > 4096:
        raise StrictTrainingError("live FA3 kernel inventory is unexpectedly broad")
    records = []
    for path in paths:
        if path.is_symlink():
            raise StrictTrainingError("live FA3 kernel inventory contains a symlink")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if expected.get("files") != records or expected.get("inventory_sha256") != digest:
        raise StrictTrainingError("live FA3 kernel bytes differ from the attention probe")
    return digest


def nanochat_effective_weight_decay(
    input_weight_decay: float,
    *,
    scaling_parameters: int,
    global_batch_tokens: int,
    d12_scaling_parameters: int,
    d12_global_batch_tokens: int = 524_288,
) -> float:
    """Reproduce upstream Nanochat's width/batch transfer for control arms."""

    values = (
        input_weight_decay,
        scaling_parameters,
        global_batch_tokens,
        d12_scaling_parameters,
        d12_global_batch_tokens,
    )
    if any(isinstance(value, bool) for value in values):
        raise StrictTrainingError("weight-decay transfer inputs are invalid")
    if input_weight_decay < 0 or min(values[1:]) <= 0:
        raise StrictTrainingError("weight-decay transfer inputs must be positive")
    return (
        float(input_weight_decay)
        * math.sqrt(global_batch_tokens / d12_global_batch_tokens)
        * (d12_scaling_parameters / scaling_parameters)
    )


def validate_recipe_invocation(
    recipe: Mapping[str, Any],
    *,
    run_kind: str,
    depth: int,
    model_config: Mapping[str, Any],
    total_parameters: int,
    scaling_parameters: int,
    world_size: int,
    device_batch_size: int,
    total_batch_size: int,
    num_iterations: int,
    stop_at_step: int,
    eval_every: int,
    seed: int,
    model_tag: str,
    lr_schedule: str,
    effective_weight_decay: float,
    weight_decay_cooldown_policy: str,
    cooldown_start_step: int | None,
) -> dict[str, Any]:
    """Bind a production, smoke, or proxy invocation to the family recipe."""

    if run_kind not in {"production", "smoke", "signal_smoke", "proxy"}:
        raise StrictTrainingError("invalid strict run kind")
    if stop_at_step <= 0 or stop_at_step > num_iterations:
        raise StrictTrainingError("stop-at-step must be inside the run horizon")
    if model_config.get("n_layer") != depth:
        raise StrictTrainingError("runtime model depth mismatch")
    if run_kind in {"production", "smoke", "signal_smoke"}:
        declared = recipe["model"]
        expected_config = {
            "sequence_len": declared["max_seq_len"],
            "vocab_size": declared["vocab_size"],
            "n_layer": declared["depth"],
            "n_head": declared["num_heads"],
            "n_kv_head": declared["num_kv_heads"],
            "n_embd": declared["model_dim"],
        }
        if any(model_config.get(field) != value for field, value in expected_config.items()):
            raise StrictTrainingError("runtime d32 model config differs from recipe")
        if total_parameters != declared["total_parameters"]:
            raise StrictTrainingError("runtime total parameter count differs from recipe")
        if scaling_parameters != declared["scaling_parameters"]:
            raise StrictTrainingError("runtime scaling parameter count differs from recipe")
        training = recipe["training"]
        if device_batch_size != training["device_batch_sequences"]:
            raise StrictTrainingError("runtime device batch differs from recipe")
        if total_batch_size != training["global_batch_tokens"]:
            raise StrictTrainingError("runtime global batch differs from recipe")
        if lr_schedule != "wsd":
            raise StrictTrainingError("d32 smoke/production runs require WSD")
        if run_kind == "signal_smoke":
            gate = recipe["distributed_gate"]
            if (
                num_iterations != gate["signal_resume_probe_updates"]
                or stop_at_step != num_iterations
                or world_size != gate["signal_resume_probe_world_size"]
            ):
                raise StrictTrainingError("signal smoke invocation differs from recipe")
            return {"recipe_scope": f"signal_smoke_ws{world_size}"}
        if run_kind == "smoke":
            if num_iterations != recipe["distributed_gate"]["smoke_updates"]:
                raise StrictTrainingError("smoke horizon differs from recipe")
            if world_size not in {8, 16}:
                raise StrictTrainingError("smoke world size must be 8 or 16")
            return {"recipe_scope": f"smoke_ws{world_size}"}
        gate = recipe["distributed_gate"]
        allowed_world_sizes = {
            gate["preferred_production_world_size"],
            gate["fallback_production_world_size"],
        }
        if world_size not in allowed_world_sizes:
            raise StrictTrainingError("production world size is outside the recipe gate")
        matching = [
            stage
            for stage in recipe["stages"]
            if stage.get("model_tag") == model_tag
            and stage.get("num_iterations") == num_iterations
            and stage.get("target_step") == stop_at_step
            and stage.get("cooldown_start_step")
            == (-1 if cooldown_start_step is None else cooldown_start_step)
        ]
        if len(matching) != 1:
            raise StrictTrainingError("production invocation does not match one recipe stage")
        selected_stage = matching[0]
        return {
            "recipe_scope": (
                "production_trunk"
                if selected_stage["kind"] == "trunk"
                else selected_stage["id"]
            ),
            "recipe_stage_id": selected_stage["id"],
        }

    proxy = recipe["weight_decay_proxy_ablation"]
    stage = proxy["screen_stage"] if depth == 12 else proxy["confirmation_stage"] if depth == 20 else None
    if stage is None:
        raise StrictTrainingError("proxy depth must be d12 or d20")
    expected_proxy_config = {
        "sequence_len": recipe["model"]["max_seq_len"],
        "vocab_size": recipe["model"]["vocab_size"],
        "n_layer": depth,
        "n_head": stage["model_dim"] // recipe["model"]["head_dim"],
        "n_kv_head": stage["model_dim"] // recipe["model"]["head_dim"],
        "n_embd": stage["model_dim"],
    }
    if any(
        model_config.get(field) != value
        for field, value in expected_proxy_config.items()
    ):
        raise StrictTrainingError("runtime proxy model config differs from recipe")
    expected = {
        "world_size": world_size,
        "device_batch_sequences": device_batch_size,
        "global_batch_tokens": total_batch_size,
        "updates": num_iterations,
        "validation_every_updates": eval_every,
    }
    for field, actual in expected.items():
        if stage[field] != actual:
            raise StrictTrainingError(f"proxy stage {field} differs from recipe")
    if seed not in stage["seeds"]:
        raise StrictTrainingError("proxy seed differs from recipe")
    if scaling_parameters != stage["scaling_parameters"]:
        raise StrictTrainingError("proxy scaling-parameter count differs from recipe")
    matches = []
    stage_key = f"d{depth}"
    for candidate in proxy["candidates"]:
        candidate_policy = (
            candidate["cooldown_weight_decay"]
            if candidate["schedule"] == "wsd"
            else "cosine_full_horizon"
        )
        candidate_start = (
            num_iterations - num_iterations // 10
            if candidate["schedule"] == "wsd"
            else None
        )
        if (
            candidate["schedule"] == lr_schedule
            and math.isclose(
                candidate["stage_effective_weight_decay"][stage_key],
                effective_weight_decay,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            and candidate_policy == weight_decay_cooldown_policy
            and candidate_start == cooldown_start_step
        ):
            matches.append(candidate)
    if len(matches) != 1:
        raise StrictTrainingError("proxy invocation does not match one recipe candidate")
    return {
        "recipe_scope": f"proxy_{stage_key}",
        "proxy_candidate_id": matches[0]["id"],
        "proxy_candidate_eligible": matches[0]["eligible_for_production"],
    }


def validate_proxy_approval(
    approval_path: str | Path,
    *,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    tokenizer_sha256: str,
    dataset_sha256: str,
    code_revision: str,
    attention_probe_sha256: str,
    production_scaling_parameters: int,
    production_global_batch_tokens: int,
    accepted_base_weight_decay: float,
    accepted_weight_decay_cooldown_policy: str,
) -> tuple[dict[str, Any], str]:
    approval = load_json_strict(approval_path)
    if not isinstance(approval, dict):
        raise StrictTrainingError("WSD proxy approval must contain an object")
    digest = verify_manifest_hash(approval)
    expected = {
        "kind": "wsd_proxy_acceptance",
        "decision": "accepted",
        "recipe_version": recipe["weight_decay_proxy_ablation"]["recipe_version"],
        "accepted_weight_decay_cooldown_policy": accepted_weight_decay_cooldown_policy,
        "weight_decay_transfer_rule": "nanochat_width_batch_v1",
        "production_scaling_parameters": production_scaling_parameters,
        "production_global_batch_tokens": production_global_batch_tokens,
        "study_manifest_sha256": recipe_sha256,
        "tokenizer_artifact_sha256": tokenizer_sha256,
        "source_dataset_manifest_sha256": dataset_sha256,
        "trainer_code_revision": code_revision,
        "attention_probe_sha256": attention_probe_sha256,
        "gradient_clip_norm": 0.0,
        "accepted_base_weight_decay": accepted_base_weight_decay,
    }
    for field, value in expected.items():
        if approval.get(field) != value:
            raise StrictTrainingError(f"WSD proxy approval mismatch at {field}")
    try:
        validate_weight_decay_proxy_candidates(
            approval.get("candidate_results"),
            production_scaling_parameters=production_scaling_parameters,
            production_global_batch_tokens=production_global_batch_tokens,
            accepted_candidate_id=approval.get("accepted_candidate_id"),
            accepted_base_weight_decay=accepted_base_weight_decay,
            accepted_weight_decay_cooldown_policy=accepted_weight_decay_cooldown_policy,
        )
    except ValueError as exc:
        raise StrictTrainingError(f"invalid WSD proxy candidate receipt: {exc}") from exc
    return approval, digest


def _strict_smoke_number(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrictTrainingError(f"{label} must be finite numeric")
    observed = float(value)
    if not math.isfinite(observed) or observed < 0 or (observed == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise StrictTrainingError(f"{label} must be {qualifier} and finite")
    return observed


def _strict_smoke_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StrictTrainingError(f"{label} must be a lowercase SHA-256")
    return value


def _fixed_smoke_receipt(
    gate_path: Path,
    *,
    world_size: int,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    preflight_sha256: str,
) -> tuple[dict[str, Any], str, list[dict[str, int]]]:
    """Reopen and semantically verify one fixed sibling smoke receipt."""

    path = gate_path.parent / f"smoke_ws{world_size}.json"
    if path.is_symlink() or not path.is_file():
        raise StrictTrainingError(
            f"fixed ws{world_size} smoke receipt is missing or unsafe: {path}"
        )
    try:
        value = load_json_strict(path)
        if not isinstance(value, dict):
            raise StrictTrainingError(
                f"fixed ws{world_size} smoke receipt must contain an object"
            )
        digest = verify_manifest_hash(value)
    except (OSError, ValueError) as exc:
        if isinstance(exc, StrictTrainingError):
            raise
        raise StrictTrainingError(
            f"invalid fixed ws{world_size} smoke receipt: {exc}"
        ) from exc

    distributed = recipe["distributed_gate"]
    gpus_per_node = int(distributed["gpus_per_node"])
    nodes = world_size // gpus_per_node
    first = int(distributed["benchmark_first_update"])
    last = int(distributed["benchmark_last_update"])
    measured_updates = last - first + 1
    scheduled_positions = measured_updates * int(recipe["training"]["global_batch_tokens"])
    expected = {
        "schema_version": "1.0",
        "kind": "d32_distributed_smoke_receipt",
        "family_id": recipe["family_id"],
        "preflight_receipt_sha256": preflight_sha256,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world_size,
        "measured_first_update": first,
        "measured_last_update": last,
        "measured_updates": measured_updates,
        "scheduled_positions": scheduled_positions,
        "packing_capacity_world_size": world_size,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise StrictTrainingError(
                f"fixed ws{world_size} smoke receipt mismatch at {field}"
            )

    completion = value.get("slurm_completion")
    slurm_job_id = value.get("slurm_job_id")
    if (
        not isinstance(slurm_job_id, str)
        or re.fullmatch(r"[0-9]+", slurm_job_id) is None
        or not isinstance(completion, Mapping)
        or completion.get("job_id") != slurm_job_id
        or completion.get("state") != "COMPLETED"
        or completion.get("exit_code") != "0:0"
        or completion.get("nodes") != nodes
    ):
        raise StrictTrainingError(
            f"fixed ws{world_size} smoke receipt lacks clean Slurm completion"
        )
    _strict_smoke_sha256(
        completion.get("sacct_output_sha256"),
        f"fixed ws{world_size} Slurm accounting output",
    )
    _strict_smoke_sha256(
        value.get("static_nccl_probe_sha256"),
        f"fixed ws{world_size} static NCCL probe",
    )
    _strict_smoke_sha256(
        value.get("static_launcher_gate_sha256"),
        f"fixed ws{world_size} static launcher gate",
    )

    identity = value.get("production_identity")
    if not isinstance(identity, Mapping) or identity.get("recipe_sha256") != recipe_sha256:
        raise StrictTrainingError(
            f"fixed ws{world_size} smoke production identity mismatch"
        )
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if value.get("production_identity_sha256") != identity_digest:
        raise StrictTrainingError(
            f"fixed ws{world_size} smoke production identity hash mismatch"
        )

    duration = _strict_smoke_number(
        value.get("duration_seconds"), f"fixed ws{world_size} smoke duration"
    )
    throughput = _strict_smoke_number(
        value.get("scheduled_positions_per_second"),
        f"fixed ws{world_size} smoke throughput",
    )
    recomputed_throughput = scheduled_positions / duration
    if not math.isclose(throughput, recomputed_throughput, rel_tol=1e-12, abs_tol=0.0):
        raise StrictTrainingError(
            f"fixed ws{world_size} smoke throughput arithmetic drifted"
        )

    loader = value.get("loader_performance")
    if not isinstance(loader, Mapping) or loader.get("passed") is not True:
        raise StrictTrainingError(
            f"fixed ws{world_size} smoke loader gate did not pass"
        )
    loader_seconds = _strict_smoke_number(
        loader.get("aggregate_loader_seconds"),
        f"fixed ws{world_size} aggregate loader seconds",
        allow_zero=True,
    )
    loader_fraction = _strict_smoke_number(
        loader.get("aggregate_loader_fraction"),
        f"fixed ws{world_size} aggregate loader fraction",
        allow_zero=True,
    )
    p95_fraction = _strict_smoke_number(
        loader.get("p95_loader_fraction"),
        f"fixed ws{world_size} p95 loader fraction",
        allow_zero=True,
    )
    maximum_aggregate = float(distributed["maximum_aggregate_loader_fraction"])
    maximum_p95 = float(distributed["maximum_p95_loader_fraction"])
    if (
        loader_seconds > duration
        or loader_fraction > maximum_aggregate
        or p95_fraction > maximum_p95
        or not math.isclose(
            loader_fraction, loader_seconds / duration, rel_tol=1e-12, abs_tol=1e-15
        )
        or loader.get("maximum_aggregate_loader_fraction") != maximum_aggregate
        or loader.get("maximum_p95_loader_fraction") != maximum_p95
    ):
        raise StrictTrainingError(
            f"fixed ws{world_size} smoke loader measurements drifted"
        )
    for field in (
        "minimum_scheduled_positions_per_second",
        "median_scheduled_positions_per_second",
    ):
        _strict_smoke_number(loader.get(field), f"fixed ws{world_size} loader {field}")

    resume = value.get("forced_resume")
    if (
        not isinstance(resume, Mapping)
        or resume.get("step") != distributed["forced_resume_step"]
        or resume.get("final_step") != distributed["smoke_updates"]
        or resume.get("verified_from_final_metadata") is not True
    ):
        raise StrictTrainingError(
            f"fixed ws{world_size} smoke forced-resume evidence drifted"
        )
    _strict_smoke_sha256(
        resume.get("checkpoint_sha256"), f"fixed ws{world_size} resume checkpoint"
    )
    _strict_smoke_sha256(
        resume.get("final_checkpoint_sha256"), f"fixed ws{world_size} final checkpoint"
    )
    _strict_smoke_sha256(
        value.get("packing_capacity_receipt_sha256"),
        f"fixed ws{world_size} packing capacity",
    )
    _strict_smoke_sha256(
        value.get("signal_resume_gate_sha256"),
        f"fixed ws{world_size} signal/resume gate",
    )
    safe_positions = value.get("packing_capacity_safe_global_scheduled_positions")
    if (
        isinstance(safe_positions, bool)
        or not isinstance(safe_positions, int)
        or safe_positions <= 0
    ):
        raise StrictTrainingError(
            f"fixed ws{world_size} packing capacity horizon must be positive"
        )

    launches = value.get("static_srun_launches")
    if not isinstance(launches, list) or [
        item.get("phase") if isinstance(item, Mapping) else None for item in launches
    ] != ["smoke_initial_50", "smoke_resume_100"]:
        raise StrictTrainingError(
            f"fixed ws{world_size} smoke launch evidence is incomplete"
        )
    for index, launch in enumerate(launches):
        assert isinstance(launch, Mapping)
        _strict_smoke_sha256(
            launch.get("sha256"), f"fixed ws{world_size} launch {index}"
        )
        if not isinstance(launch.get("path"), str) or not launch["path"]:
            raise StrictTrainingError(
                f"fixed ws{world_size} launch {index} path is missing"
            )

    storage = value.get("checkpoint_storage")
    if not isinstance(storage, Mapping) or set(storage) != {"forced_resume", "final"}:
        raise StrictTrainingError(
            f"fixed ws{world_size} smoke checkpoint storage is incomplete"
        )
    storage_records: list[dict[str, int]] = []
    for boundary in ("forced_resume", "final"):
        record = storage.get(boundary)
        if not isinstance(record, Mapping):
            raise StrictTrainingError(
                f"fixed ws{world_size} {boundary} storage must be an object"
            )
        observed: dict[str, int] = {}
        for field in (
            "full_transaction_bytes",
            "declared_payload_bytes",
            "model_metadata_completion_bytes",
        ):
            raw = record.get(field)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise StrictTrainingError(
                    f"fixed ws{world_size} {boundary} {field} must be positive"
                )
            observed[field] = raw
        if (
            observed["declared_payload_bytes"] >= observed["full_transaction_bytes"]
            or observed["model_metadata_completion_bytes"]
            > observed["full_transaction_bytes"]
        ):
            raise StrictTrainingError(
                f"fixed ws{world_size} {boundary} storage arithmetic drifted"
            )
        storage_records.append(observed)

    curve_log = value.get("curve_log")
    if (
        not isinstance(curve_log, Mapping)
        or not isinstance(curve_log.get("path"), str)
        or not curve_log["path"]
        or isinstance(curve_log.get("event_count"), bool)
        or not isinstance(curve_log.get("event_count"), int)
        or curve_log["event_count"] < measured_updates
    ):
        raise StrictTrainingError(
            f"fixed ws{world_size} smoke curve-log evidence is incomplete"
        )
    _strict_smoke_sha256(
        curve_log.get("sha256"), f"fixed ws{world_size} smoke curve log"
    )
    _strict_smoke_sha256(
        curve_log.get("last_event_sha256"),
        f"fixed ws{world_size} smoke curve-log tail",
    )
    return value, digest, storage_records


def _recompute_topology_smoke_evidence(
    gate_path: str | Path,
    gate: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    """Recompute the topology decision from fixed sibling smoke receipts."""

    path = Path(gate_path)
    smoke8, smoke8_sha, storage_records = _fixed_smoke_receipt(
        path,
        world_size=8,
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        preflight_sha256=preflight_sha256,
    )
    if gate.get("smoke_8gpu_sha256") != smoke8_sha:
        raise StrictTrainingError("production gate ws8 smoke hash differs from fixed receipt")

    smoke16_path = path.parent / "smoke_ws16.json"
    claimed_smoke16_sha = gate.get("smoke_16gpu_sha256")
    smoke16 = None
    smoke16_sha = None
    if claimed_smoke16_sha is None:
        if smoke16_path.exists() or smoke16_path.is_symlink():
            raise StrictTrainingError(
                "fixed ws16 smoke receipt exists but the production gate omitted it"
            )
    else:
        smoke16, smoke16_sha, smoke16_storage = _fixed_smoke_receipt(
            path,
            world_size=16,
            recipe=recipe,
            recipe_sha256=recipe_sha256,
            preflight_sha256=preflight_sha256,
        )
        storage_records.extend(smoke16_storage)
        if claimed_smoke16_sha != smoke16_sha:
            raise StrictTrainingError(
                "production gate ws16 smoke hash differs from fixed receipt"
            )

    identity8 = smoke8["production_identity"]
    if smoke16 is not None and (
        smoke16.get("production_identity") != identity8
        or smoke16.get("production_identity_sha256")
        != smoke8.get("production_identity_sha256")
        or smoke16.get("signal_resume_gate_sha256")
        != smoke8.get("signal_resume_gate_sha256")
        or smoke16.get("static_launcher_gate_sha256")
        != smoke8.get("static_launcher_gate_sha256")
        or smoke16.get("packing_capacity_receipt_sha256")
        != smoke8.get("packing_capacity_receipt_sha256")
    ):
        raise StrictTrainingError("fixed ws8/ws16 smoke evidence has mixed lineage")

    throughput8 = float(smoke8["scheduled_positions_per_second"])
    throughput16 = (
        None
        if smoke16 is None
        else float(smoke16["scheduled_positions_per_second"])
    )
    speedup = None if throughput16 is None else throughput16 / throughput8
    threshold = float(recipe["distributed_gate"]["minimum_8_to_16_gpu_speedup"])
    preferred = speedup is not None and speedup >= threshold
    selected_world_size = 16 if preferred else 8
    gpus_per_node = int(recipe["distributed_gate"]["gpus_per_node"])
    selected_nodes = selected_world_size // gpus_per_node
    selected_smoke = smoke16 if preferred else smoke8
    assert selected_smoke is not None
    if smoke16 is None:
        selection_reason = "no_clean_16gpu_smoke_receipt_supplied_use_8gpu_fallback"
    elif preferred:
        selection_reason = "clean_16gpu_smoke_meets_minimum_1.7_speedup"
    else:
        selection_reason = "clean_16gpu_smoke_below_minimum_1.7_speedup_use_8gpu_fallback"

    storage_factor = float(recipe["storage"]["smoke_measurement_safety_factor"])
    measured_full = max(record["full_transaction_bytes"] for record in storage_records)
    measured_model = max(
        record["model_metadata_completion_bytes"] for record in storage_records
    )
    storage_calibration = {
        "safety_factor": storage_factor,
        "maximum_measured_full_transaction_bytes": measured_full,
        "maximum_measured_model_bundle_bytes": measured_model,
        "calibrated_full_transaction_bytes": math.ceil(measured_full * storage_factor),
        "calibrated_model_bundle_bytes": math.ceil(measured_model * storage_factor),
    }
    return {
        "smoke_8gpu_sha256": smoke8_sha,
        "smoke_16gpu_sha256": smoke16_sha,
        "throughput_8gpu": throughput8,
        "throughput_16gpu": throughput16,
        "speedup_8_to_16": speedup,
        "parallel_efficiency": None if speedup is None else speedup / 2.0,
        "preferred_topology_accepted": preferred,
        "selection_reason": selection_reason,
        "authorized_production_world_size": selected_world_size,
        "authorized_production_nodes": selected_nodes,
        "authorized_safe_global_scheduled_positions": selected_smoke[
            "packing_capacity_safe_global_scheduled_positions"
        ],
        "selected_smoke_sha256": smoke16_sha if preferred else smoke8_sha,
        "selected_throughput": throughput16 if preferred else throughput8,
        "packing_capacity_receipt_sha256": smoke8[
            "packing_capacity_receipt_sha256"
        ],
        "signal_resume_gate_sha256": smoke8["signal_resume_gate_sha256"],
        "production_identity": identity8,
        "storage_calibration": storage_calibration,
    }


def validate_production_topology_gate(
    gate_path: str | Path,
    preflight_receipt_path: str | Path,
    *,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    attention_probe_sha256: str,
    proxy_approval_sha256: str,
    accepted_base_weight_decay: float,
    accepted_weight_decay_cooldown_policy: str,
    world_size: int,
    packing_capacity_receipt_sha256: str,
    selected_capacity: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Verify the sealed smoke/signal decision authorizing one production WS."""

    preflight = load_json_strict(preflight_receipt_path)
    if not isinstance(preflight, dict):
        raise StrictTrainingError("production topology preflight must be an object")
    try:
        preflight_sha = verify_manifest_hash(preflight)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid topology preflight receipt: {exc}") from exc
    if (
        preflight.get("kind") != "d32_family_preflight_receipt"
        or preflight.get("recipe", {}).get("canonical_sha256") != recipe_sha256
    ):
        raise StrictTrainingError("production topology preflight identity mismatch")
    gate = load_json_strict(gate_path)
    if not isinstance(gate, dict):
        raise StrictTrainingError("production topology gate must contain an object")
    try:
        gate_sha = verify_manifest_hash(gate)
    except ValueError as exc:
        raise StrictTrainingError(f"invalid production topology gate: {exc}") from exc
    smoke_evidence = _recompute_topology_smoke_evidence(
        gate_path,
        gate,
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        preflight_sha256=preflight_sha,
    )
    for field in (
        "smoke_8gpu_sha256",
        "smoke_16gpu_sha256",
        "throughput_8gpu",
        "throughput_16gpu",
        "speedup_8_to_16",
        "parallel_efficiency",
        "preferred_topology_accepted",
        "selection_reason",
        "authorized_production_world_size",
        "authorized_production_nodes",
        "authorized_safe_global_scheduled_positions",
        "packing_capacity_receipt_sha256",
        "signal_resume_gate_sha256",
        "storage_calibration",
    ):
        if gate.get(field) != smoke_evidence[field]:
            raise StrictTrainingError(
                f"production topology gate drifted from fixed smoke evidence at {field}"
            )
    smoke_identity = smoke_evidence["production_identity"]
    expected_smoke_identity = {
        "recipe_sha256": recipe_sha256,
        "attention_probe_sha256": attention_probe_sha256,
        "wsd_proxy_approval_sha256": proxy_approval_sha256,
        "wsd_base_weight_decay": accepted_base_weight_decay,
        "wsd_weight_decay_cooldown": accepted_weight_decay_cooldown_policy,
    }
    for field, wanted in expected_smoke_identity.items():
        if smoke_identity.get(field) != wanted:
            raise StrictTrainingError(
                f"fixed smoke production identity mismatch at {field}"
            )
    preflight_corpus = preflight.get("corpus")
    preflight_capacity = (
        preflight_corpus.get("packing_capacity_receipt")
        if isinstance(preflight_corpus, Mapping)
        else None
    )
    if not isinstance(preflight_capacity, Mapping):
        raise StrictTrainingError("production preflight lacks packing-capacity evidence")
    preflight_worlds = preflight_capacity.get("worlds")
    preflight_world = (
        preflight_worlds.get(str(world_size))
        if isinstance(preflight_worlds, Mapping)
        else None
    )
    try:
        safe_positions = capacity_authorized_positions(selected_capacity)
        preflight_safe_positions = (
            capacity_authorized_positions(preflight_world)
            if isinstance(preflight_world, Mapping)
            else None
        )
    except StrictTrainingError as exc:
        raise StrictTrainingError(
            "production topology gate capacity differs from the sealed preflight"
        ) from exc
    if recipe["family_id"] in _REPEAT_CAPACITY_FAMILY_IDS:
        selected_capacity_matches_preflight = (
            preflight_world == selected_capacity
            and selected_capacity.get("capacity_mode") == "whole_pool_repeat_v3"
        )
    else:
        selected_capacity_matches_preflight = (
            isinstance(preflight_world, Mapping)
            and preflight_world.get("passes_40x_no_wrap_with_margin") is True
            and preflight_safe_positions == safe_positions
        )
    if (
        not isinstance(packing_capacity_receipt_sha256, str)
        or len(packing_capacity_receipt_sha256) != 64
        or not isinstance(preflight_world, Mapping)
        or preflight_capacity.get("sha256") != packing_capacity_receipt_sha256
        or not selected_capacity_matches_preflight
    ):
        raise StrictTrainingError(
            "production topology gate capacity differs from the sealed preflight"
        )
    expected = {
        "kind": "d32_production_topology_gate",
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha256,
        "preflight_receipt_sha256": preflight_sha,
        "attention_probe_sha256": attention_probe_sha256,
        "wsd_proxy_approval_sha256": proxy_approval_sha256,
        "accepted_base_weight_decay": accepted_base_weight_decay,
        "accepted_weight_decay_cooldown_policy": (
            accepted_weight_decay_cooldown_policy
        ),
        "passed": True,
        "authorized_production_world_size": world_size,
        "packing_capacity_receipt_sha256": packing_capacity_receipt_sha256,
        "authorized_packing_capacity_world_size": world_size,
        "authorized_safe_global_scheduled_positions": safe_positions,
        "require_single_world_size_for_entire_lineage": True,
    }
    for field, wanted in expected.items():
        if gate.get(field) != wanted:
            raise StrictTrainingError(f"production topology gate mismatch at {field}")
    distributed = recipe["distributed_gate"]
    expected_nodes = 4 if world_size == 16 else 2 if world_size == 8 else None
    if (
        expected_nodes is None
        or gate.get("authorized_production_nodes") != expected_nodes
        or gate.get("required_speedup")
        != distributed["minimum_8_to_16_gpu_speedup"]
    ):
        raise StrictTrainingError("production topology node/speedup policy mismatch")
    if not isinstance(gate.get("signal_resume_gate_sha256"), str) or len(
        gate["signal_resume_gate_sha256"]
    ) != 64:
        raise StrictTrainingError("production gate lacks signal-resume evidence")
    if not isinstance(gate.get("smoke_8gpu_sha256"), str) or len(
        gate["smoke_8gpu_sha256"]
    ) != 64:
        raise StrictTrainingError("production gate lacks the mandatory ws8 smoke")
    if world_size == 16:
        if (
            not isinstance(gate.get("smoke_16gpu_sha256"), str)
            or len(gate["smoke_16gpu_sha256"]) != 64
            or gate.get("preferred_topology_accepted") is not True
            or float(gate.get("speedup_8_to_16", 0.0))
            < float(distributed["minimum_8_to_16_gpu_speedup"])
            or float(gate.get("parallel_efficiency", 0.0))
            < float(distributed["minimum_parallel_efficiency"])
        ):
            raise StrictTrainingError("ws16 topology lacks its throughput acceptance")
    elif gate.get("preferred_topology_accepted") is True:
        raise StrictTrainingError("ws8 was authorized despite accepting preferred ws16")
    if not isinstance(gate.get("selection_reason"), str) or not gate["selection_reason"]:
        raise StrictTrainingError("production topology gate lacks a selection reason")
    if not isinstance(gate.get("storage_calibration"), Mapping):
        raise StrictTrainingError("production topology gate lacks storage calibration")
    selected_smoke_sha = smoke_evidence["selected_smoke_sha256"]
    selected_throughput = smoke_evidence["selected_throughput"]
    try:
        throughput = float(selected_throughput)
        nodes = int(gate["authorized_production_nodes"])
        stage_updates = sum(
            int(stage["target_step"]) - int(stage.get("source_step") or 0)
            for stage in recipe["stages"]
        )
        full_positions = stage_updates * int(recipe["training"]["global_batch_tokens"])
        budget = recipe["uhem_budget"]
        billing_rate = int(budget["cpu_saat_per_4gpu_node_hour"])
        raw_cost = full_positions / throughput / 3600.0 * nodes * billing_rate
        raw_ceiling = math.ceil(raw_cost)
        reserved = math.ceil(raw_cost * 1.15)
        allowance = int(budget["proxy_and_smoke_reserve_cpu_saat"])
        operational_ceiling = int(budget["operational_ceiling_cpu_saat"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise StrictTrainingError(
            f"production topology measured-cost inputs are invalid: {exc}"
        ) from exc
    if not math.isfinite(throughput) or throughput <= 0:
        raise StrictTrainingError("production topology measured throughput is invalid")
    expected_cost = {
        "version": "measured_smoke_v1",
        "selected_smoke_sha256": selected_smoke_sha,
        "world_size": world_size,
        "nodes": nodes,
        "global_batch_tokens": int(recipe["training"]["global_batch_tokens"]),
        "full_shared_updates": stage_updates,
        "full_scheduled_positions": full_positions,
        "measured_positions_per_second": throughput,
        "billing_cpu_saat_per_node_hour": billing_rate,
        "reserve_fraction": 0.15,
        "raw_training_cpu_saat_ceiling": raw_ceiling,
        "reserved_training_cpu_saat": reserved,
        "proxy_smoke_allowance_cpu_saat": allowance,
        "projected_total_package_cpu_saat": reserved + allowance,
        "operational_ceiling_cpu_saat": operational_ceiling,
        "passed": reserved + allowance <= operational_ceiling,
    }
    if gate.get("cost_projection") != expected_cost or expected_cost["passed"] is not True:
        raise StrictTrainingError(
            "production topology measured-cost projection is invalid or over budget"
        )
    return gate, gate_sha


__all__ = [
    "ArtifactBindings",
    "FAMILY_ARTIFACT_CONTRACTS",
    "FAMILY_ID",
    "FAMILY_ID_V2",
    "FAMILY_ID_V3",
    "FAMILY_ID_V4",
    "PINNED_UPSTREAM_REVISION",
    "PREEMPTION_EXIT_CODE",
    "SeedPlan",
    "StrictTrainingError",
    "capacity_authorized_positions",
    "capacity_world_gate_record",
    "derive_seed_plan",
    "file_sha256",
    "family_artifact_contract",
    "load_artifact_bindings",
    "nanochat_effective_weight_decay",
    "validate_attention_probe_receipt",
    "validate_bestfit_capacity_receipt",
    "validate_family_recipe",
    "validate_preflight_artifact_bindings",
    "validate_proxy_approval",
    "validate_production_topology_gate",
    "validate_recipe_invocation",
    "verify_live_fa3_kernel_inventory",
    "verify_code_provenance",
]
