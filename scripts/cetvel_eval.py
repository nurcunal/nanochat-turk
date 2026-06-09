"""
Run CETVEL benchmark suites on a trained nanochat base model.

CETVEL is distributed as lm-evaluation-harness tasks. This script keeps the
benchmark definitions in CETVEL and only registers nanochat as an lm-eval model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict

from nanochat.common import get_base_dir, print0


CETVEL_SUITES = {
    # Cheap, representative iteration suite. Use --limit during debugging.
    "fast": [
        "belebele_tr",
        "cetvel_xnli_tr",
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
        "cetvel_xcopa_tr",
        "cetvel_xnli_tr",
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
        "cetvel_xcopa_tr",
        "cetvel_xnli_tr",
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
        "gecturk_generation",
    ],
}


HF_DATASET_ALIASES = {
    # CETVEL is pinned to older lm-eval task configs. Current
    # huggingface_hub rejects one-part dataset IDs in hf:// URIs for several
    # legacy aliases, so rewrite them to their canonical Hub repos locally.
    "exams": "mhardalov/exams",
    "nli_tr": "boun-tabi/nli_tr",
    "offenseval2020_tr": "coltekin/offenseval2020_tr",
    "xcopa": "cambridgeltl/xcopa",
    "xnli": "facebook/xnli",
    "xquad": "google/xquad",
    "xfact": "utahnlp/x-fact",
    "mlsum": "reciTAL/mlsum",
    "xlsum": "csebuetnlp/xlsum",
    "wiki_lingua": "GEM/wiki_lingua",
    "wmt16": "wmt/wmt16",
}


CETVEL_TASK_RENAMES = {
    # Avoid name collisions where lm-eval's built-in registry wins over
    # include_path tasks with the same name.
    "xcopa": "cetvel_xcopa",
    "xcopa_et": "cetvel_xcopa_et",
    "xcopa_tr": "cetvel_xcopa_tr",
    "xnli_tr": "cetvel_xnli_tr",
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


def _assert_datasets_supports_legacy_scripts() -> None:
    try:
        import datasets
    except Exception:
        return

    major = int(datasets.__version__.split(".", 1)[0])
    if major >= 4:
        raise RuntimeError(
            "CETVEL's Turkish NLI tasks still depend on a Hugging Face dataset script. "
            "Install datasets<4 for CETVEL, for example: python -m pip install 'datasets==2.19.2'."
        )


def _patch_cetvel_task_configs(cetvel_dir: str) -> None:
    tasks_dir = os.path.join(cetvel_dir, "tasks")
    if not os.path.isdir(tasks_dir):
        return

    def replace_alias(match: re.Match[str], canonical: str) -> str:
        suffix = match.group(3) if match.lastindex and match.lastindex >= 3 else ""
        return f"{match.group(1)}{canonical}{suffix}"

    patched: list[str] = []
    for root, _dirs, files in os.walk(tasks_dir):
        for name in files:
            if not name.endswith((".yaml", ".yml", "_yaml", ".py")):
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            new_text = text
            for alias, canonical in HF_DATASET_ALIASES.items():
                escaped = re.escape(alias)
                replacements = [
                    re.compile(
                        rf"(?m)^(\s*(?:dataset_path|path)\s*:\s*)(['\"]?){escaped}\2(\s*(?:#.*)?)$"
                    ),
                    re.compile(rf"(\b(?:dataset_path|path)\s*:\s*)(['\"]?){escaped}\2(?=\s|$)"),
                ]
                for pattern in replacements:
                    new_text = pattern.sub(lambda match, value=canonical: replace_alias(match, value), new_text)
                py_replacements = [
                    re.compile(rf"(?m)^(\s*DATASET_PATH\s*=\s*)(['\"]){escaped}\2(\s*(?:#.*)?)$"),
                    re.compile(rf"(datasets\.load_dataset\(\s*)(['\"]){escaped}\2"),
                ]
                for pattern in py_replacements:
                    new_text = pattern.sub(
                        lambda match, value=canonical: f"{match.group(1)}{match.group(2)}{value}{match.group(2)}"
                        + (match.group(3) if match.lastindex and match.lastindex >= 3 else ""),
                        new_text,
                    )
            for old_task, new_task in CETVEL_TASK_RENAMES.items():
                escaped = re.escape(old_task)
                task_patterns = [
                    re.compile(rf"(?m)^(\s*(?:task|group)\s*:\s*)(['\"]?){escaped}\2(\s*(?:#.*)?)$"),
                    re.compile(rf"(\b(?:task|group)\s*:\s*)(['\"]?){escaped}\2(?=\s|$)"),
                ]
                for pattern in task_patterns:
                    new_text = pattern.sub(lambda match, value=new_task: replace_alias(match, value), new_text)
            if new_text != text:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)
                patched.append(path)
    for path in patched:
        print0(f"Patched CETVEL task config for current HF dataset aliases: {path}")


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            _flatten(key, v, out)
    elif isinstance(value, (int, float)):
        out[prefix] = value


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        elif isinstance(value, list) and isinstance(dst.get(key), list):
            dst[key].extend(value)
        else:
            dst[key] = value
    return dst


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return name or "task"


def _write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def _resolve_tasks(args) -> list[str]:
    if args.tasks:
        return [task.strip() for task in args.tasks.split(",") if task.strip()]
    return CETVEL_SUITES[args.suite]


def _run_lm_eval(
    evaluator,
    model_args_str: str,
    tasks: list[str],
    args,
    tracker,
    task_manager,
):
    return evaluator.simple_evaluate(
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


def _evaluate_with_task_progress(
    evaluator,
    evaluation_tracker_cls,
    model_args_str: str,
    tasks: list[str],
    args,
    out_dir: str,
    task_manager,
    wandb_run=None,
) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    task_results_dir = os.path.join(out_dir, "task_results")
    os.makedirs(task_results_dir, exist_ok=True)
    suite_started = time.monotonic()

    print0(
        f"[CETVEL progress] running {len(tasks)} task/group evaluations one at a time",
        flush=True,
    )
    for index, task in enumerate(tasks, start=1):
        task_started = time.monotonic()
        task_slug = _safe_filename(task)
        task_out_dir = os.path.join(task_results_dir, f"{index:02d}_{task_slug}")
        os.makedirs(task_out_dir, exist_ok=True)

        print0(
            f"[CETVEL progress] {index}/{len(tasks)} START {task}",
            flush=True,
        )
        task_results = _run_lm_eval(
            evaluator=evaluator,
            model_args_str=model_args_str,
            tasks=[task],
            args=args,
            tracker=evaluation_tracker_cls(output_path=task_out_dir),
            task_manager=task_manager,
        )
        _deep_merge(aggregate, task_results)

        task_results_path = os.path.join(task_out_dir, f"cetvel_{args.suite}_{task_slug}_results.json")
        partial_results_path = os.path.join(out_dir, f"cetvel_{args.suite}_partial_results.json")
        _write_json(task_results_path, task_results)
        _write_json(partial_results_path, aggregate)

        task_elapsed_sec = time.monotonic() - task_started
        suite_elapsed_sec = time.monotonic() - suite_started
        task_elapsed = _format_duration(task_elapsed_sec)
        suite_elapsed = _format_duration(suite_elapsed_sec)
        print0(
            f"[CETVEL progress] {index}/{len(tasks)} DONE {task} "
            f"task_elapsed={task_elapsed} total_elapsed={suite_elapsed} "
            f"partial={partial_results_path}",
            flush=True,
        )
        if wandb_run is not None:
            task_metrics: Dict[str, Any] = {
                "cetvel_progress/tasks_done": index,
                "cetvel_progress/tasks_total": len(tasks),
                "cetvel_progress/task_elapsed_sec": task_elapsed_sec,
                "cetvel_progress/total_elapsed_sec": suite_elapsed_sec,
            }
            if isinstance(task_results, dict):
                _flatten(f"cetvel_task/{task}", task_results.get("results", {}), task_metrics)
                _flatten(f"cetvel_task_groups/{task}", task_results.get("groups", {}), task_metrics)
            wandb_run.log(task_metrics)

    return aggregate


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
    parser.add_argument(
        "--task-progress",
        action="store_true",
        help="Evaluate tasks/groups one at a time and print progress with partial result files.",
    )
    parser.add_argument("--patch-configs-only", action="store_true", help="Patch local CETVEL task configs and exit.")
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
    _patch_cetvel_task_configs(cetvel_dir)

    if args.patch_configs_only:
        return

    if not os.path.isdir(include_path):
        print0(
            "CETVEL tasks directory not found.\n"
            f"Expected: {include_path}\n"
            "Run with --auto-setup, or clone CETVEL with submodules and pass --cetvel-dir."
        )
        return

    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(base_dir, "cetvel_hf_datasets_cache_datasets219"))
    _assert_datasets_supports_legacy_scripts()

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
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "nanochat-turk"),
            name=os.environ.get("WANDB_RUN", f"cetvel-{args.suite}"),
            job_type="cetvel",
            reinit=True,
            config={
                "suite": args.suite,
                "tasks": tasks,
                "model_tag": args.model_tag,
                "model_step": args.model_step,
                "batch_size": args.batch_size,
                "device": args.device,
                "max_gen_tokens": args.max_gen_tokens,
                "task_progress": args.task_progress,
            },
        )
        wandb_run.log({
            "cetvel_progress/tasks_done": 0,
            "cetvel_progress/tasks_total": len(tasks),
        })

    try:
        if args.task_progress and len(tasks) > 1:
            results = _evaluate_with_task_progress(
                evaluator=evaluator,
                evaluation_tracker_cls=EvaluationTracker,
                model_args_str=model_args_str,
                tasks=tasks,
                args=args,
                out_dir=out_dir,
                task_manager=task_manager,
                wandb_run=wandb_run,
            )
        else:
            results = _run_lm_eval(
                evaluator=evaluator,
                model_args_str=model_args_str,
                tasks=tasks,
                args=args,
                tracker=EvaluationTracker(output_path=out_dir),
                task_manager=task_manager,
            )
    except KeyError as exc:
        available = sorted(task_manager.all_tasks)
        print0(f"Task not found: {exc}")
        print0("Available CETVEL tasks/groups include:")
        print0(", ".join(available[:200]))
        raise

    results_path = os.path.join(out_dir, f"cetvel_{args.suite}_results.json")
    _write_json(results_path, results)
    print0(f"Wrote CETVEL results to {results_path}")

    metrics: Dict[str, Any] = {}
    if isinstance(results, dict):
        _flatten("cetvel", results.get("results", {}), metrics)
        _flatten("cetvel_groups", results.get("groups", {}), metrics)

    if wandb_run is not None:
        if metrics:
            wandb_run.log(metrics)
        import wandb
        wandb.save(results_path)
        wandb_run.finish()

    from nanochat.report import get_report
    get_report().log(section=f"CETVEL {args.suite}", data=[
        {"tasks": tasks, "results_path": results_path},
        metrics,
    ])


if __name__ == "__main__":
    main()
