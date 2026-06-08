"""
Run CETVEL benchmark suites on a trained nanochat base model.

CETVEL is distributed as lm-evaluation-harness tasks. This script keeps the
benchmark definitions in CETVEL and only registers nanochat as an lm-eval model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict

from nanochat.common import get_base_dir, print0


CETVEL_SUITES = {
    # Cheap, representative iteration suite. Use --limit during debugging.
    "fast": [
        "belebele_tr",
        "xnli_tr",
        "news_cat",
        "xquad_tr",
        "turkish_plu_next_event_prediction",
        "exams_tr",
    ],
    # Major foundation checkpoint suite: understanding-heavy and useful for
    # base models, before instruction tuning.
    "core": [
        "exams_tr",
        "belebele_tr",
        "turkish_plu",
        "xcopa_tr",
        "xnli_tr",
        "mnli_tr",
        "snli_tr",
        "news_cat",
        "offenseval_tr",
        "trclaim19",
        "xfact_tr",
        "xquad_tr",
        "tquad",
        "mkqa_tr",
    ],
    # Full CETVEL adds generation-heavy tasks. These are useful diagnostics for
    # base models, but become more meaningful after SFT.
    "full": [
        "exams_tr",
        "belebele_tr",
        "turkish_plu",
        "xcopa_tr",
        "xnli_tr",
        "mnli_tr",
        "snli_tr",
        "news_cat",
        "offenseval_tr",
        "trclaim19",
        "xfact_tr",
        "xquad_tr",
        "tquad",
        "mkqa_tr",
        "wmt-tr-en-prompt",
        "wmt-en-tr-prompt",
        "mlsum_tr",
        "xlsum_tr",
        "wiki_lingua_tr",
        "gecturk_gen",
    ],
}


def _maybe_setup_cetvel(cetvel_dir: str) -> None:
    if not os.path.isdir(cetvel_dir):
        print0(f"Cloning CETVEL into {cetvel_dir} ...")
        subprocess.check_call([
            "git", "clone", "--depth", "1", "--recurse-submodules",
            "https://github.com/KUIS-AI/cetvel.git", cetvel_dir,
        ])
    else:
        subprocess.check_call(["git", "-C", cetvel_dir, "submodule", "update", "--init", "--recursive"])

    harness_dir = os.path.join(cetvel_dir, "lm-evaluation-harness")
    req_path = os.path.join(cetvel_dir, "requirements.txt")
    _pip_install(["toml"])
    _pip_install(["-e", harness_dir])
    if os.path.isfile(req_path):
        _pip_install(["-r", req_path])
    _prepend_pythonpath(harness_dir)


def _pip_install(args: list[str]) -> None:
    py = sys.executable
    try:
        subprocess.check_call(
            [py, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError(
                f"{py} has no pip module and uv is not available; cannot install CETVEL dependencies."
            )
        subprocess.check_call([uv, "pip", "install", "-q", *args])
        return

    subprocess.check_call([py, "-m", "pip", "install", "-q", *args])


def _prepend_pythonpath(path: str) -> None:
    if not os.path.isdir(path):
        return
    if path not in sys.path:
        sys.path.insert(0, path)
    paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if path not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([path, *paths])


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            _flatten(key, v, out)
    elif isinstance(value, (int, float)):
        out[prefix] = value


def _resolve_tasks(args) -> list[str]:
    if args.tasks:
        return [task.strip() for task in args.tasks.split(",") if task.strip()]
    return CETVEL_SUITES[args.suite]


def main() -> None:
    parser = argparse.ArgumentParser(description="CETVEL evaluation for nanochat")
    parser.add_argument("--suite", type=str, default="fast", choices=sorted(CETVEL_SUITES))
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated CETVEL tasks/groups. Overrides --suite.")
    parser.add_argument("--list-suites", action="store_true", help="Print suite task lists and exit.")
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--model-step", type=int, default=None)
    parser.add_argument("--cetvel-dir", type=str, default=os.environ.get("CETVEL_DIR", ""))
    parser.add_argument("--limit", type=float, default=None, help="lm-eval --limit, useful for quick checks.")
    parser.add_argument("--batch-size", type=str, default="1")
    parser.add_argument("--device", type=str, default=os.environ.get("CETVEL_DEVICE", "cuda:0"))
    parser.add_argument("--max-gen-tokens", type=int, default=128)
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--auto-setup", action="store_true", help="Clone CETVEL and install lm-eval harness if needed.")
    parser.add_argument("--wandb", action="store_true", help="Log flattened numeric metrics to wandb.")
    args = parser.parse_args()

    if args.list_suites:
        print(json.dumps(CETVEL_SUITES, indent=2, ensure_ascii=False))
        return

    base_dir = get_base_dir()
    cetvel_dir = os.path.abspath(args.cetvel_dir or os.path.join(base_dir, "cetvel"))
    include_path = os.path.join(cetvel_dir, "tasks")
    harness_dir = os.path.join(cetvel_dir, "lm-evaluation-harness")

    if args.auto_setup:
        _maybe_setup_cetvel(cetvel_dir)
    else:
        _prepend_pythonpath(harness_dir)

    if not os.path.isdir(include_path):
        print0(
            "CETVEL tasks directory not found.\n"
            f"Expected: {include_path}\n"
            "Run with --auto-setup, or clone CETVEL with submodules and pass --cetvel-dir."
        )
        return

    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(base_dir, "cetvel_hf_datasets_cache"))

    try:
        from lm_eval import evaluator
        from lm_eval.loggers import EvaluationTracker
        from lm_eval.tasks import TaskManager
    except Exception as exc:
        if args.auto_setup:
            _maybe_setup_cetvel(cetvel_dir)
            from lm_eval import evaluator
            from lm_eval.loggers import EvaluationTracker
            from lm_eval.tasks import TaskManager
        else:
            raise RuntimeError("lm-evaluation-harness is not installed. Re-run with --auto-setup.") from exc

    import nanochat.lm_eval_nanochat  # noqa: F401

    tasks = _resolve_tasks(args)
    out_dir = args.output_path or os.path.join(base_dir, "cetvel_out", args.suite)
    os.makedirs(out_dir, exist_ok=True)

    model_args = [f"base_dir={base_dir}", f"max_gen_tokens={args.max_gen_tokens}"]
    if args.model_tag:
        model_args.append(f"model_tag={args.model_tag}")
    if args.model_step is not None:
        model_args.append(f"model_step={args.model_step}")
    model_args_str = ",".join(model_args)

    print0("Running CETVEL via lm-evaluation-harness:")
    print0(f"- suite: {args.suite}")
    print0(f"- include_path: {include_path}")
    print0(f"- tasks: {','.join(tasks)}")
    print0(f"- model_args: {model_args_str}")

    task_manager = TaskManager("INFO", include_path=include_path)
    tracker = EvaluationTracker(output_path=out_dir)

    try:
        results = evaluator.simple_evaluate(
            model="nanochat",
            model_args=model_args_str,
            tasks=tasks,
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
            write_out=True,
            log_samples=True,
            evaluation_tracker=tracker,
            task_manager=task_manager,
        )
    except KeyError as exc:
        available = sorted(task_manager.all_tasks)
        print0(f"Task not found: {exc}")
        print0("Available CETVEL tasks/groups include:")
        print0(", ".join(available[:200]))
        raise

    results_path = os.path.join(out_dir, f"cetvel_{args.suite}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print0(f"Wrote CETVEL results to {results_path}")

    metrics: Dict[str, Any] = {}
    if isinstance(results, dict):
        _flatten("cetvel", results.get("results", {}), metrics)
        _flatten("cetvel_groups", results.get("groups", {}), metrics)

    if args.wandb and metrics:
        import wandb
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "nanochat-turk"),
            name=os.environ.get("WANDB_RUN", f"cetvel-{args.suite}"),
            job_type="cetvel",
            reinit=True,
        )
        wandb.log(metrics)
        wandb.save(results_path)
        run.finish()

    from nanochat.report import get_report
    get_report().log(section=f"CETVEL {args.suite}", data=[
        {"tasks": tasks, "results_path": results_path},
        metrics,
    ])


if __name__ == "__main__":
    main()
