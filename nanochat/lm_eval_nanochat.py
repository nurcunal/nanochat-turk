"""
lm-evaluation-harness model adapter for nanochat checkpoints.

This lets CETVEL evaluate nanochat base checkpoints through the standard
lm-eval task interface.
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import torch

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from nanochat.checkpoint_manager import load_model
from nanochat.common import get_base_dir, print0
from nanochat.engine import Engine


@register_model("nanochat")
class NanochatLM(LM):
    """
    Minimal adapter:
    - loglikelihood for multiple-choice and classification tasks
    - generate_until for generation tasks

    model_args example:
      base_dir=/path/to/out,model_tag=tr_d24,model_step=1000,max_gen_tokens=128
    """

    def __init__(
        self,
        base_dir: str | None = None,
        model_tag: str | None = None,
        model_step: int | None = None,
        device: str | None = None,
        batch_size: str | int | None = None,
        max_batch_size: int | None = None,
        max_gen_tokens: int = 128,
        **_: object,
    ) -> None:
        super().__init__()
        if base_dir:
            os.environ["NANOCHAT_BASE_DIR"] = base_dir

        self.base_dir = get_base_dir()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_gen_tokens = int(max_gen_tokens)

        model, tokenizer, meta = load_model(
            "base",
            device=self.device,
            phase="eval",
            model_tag=model_tag,
            step=model_step,
        )
        self.model = model
        if self.device.type == "cuda":
            self.model = self.model.to(dtype=torch.bfloat16)
        self.tokenizer = tokenizer
        self.engine = Engine(self.model, self.tokenizer)
        self.model_max_len = int(meta["model_config"]["sequence_len"])
        self.bos_id = int(tokenizer.get_bos_token_id())

        # lm-eval loggers expect a few HuggingFace-like tokenizer attributes.
        for name, value in {
            "pad_token": "<|bos|>",
            "pad_token_id": self.bos_id,
            "bos_token": "<|bos|>",
            "bos_token_id": self.bos_id,
            "eos_token": "<|bos|>",
            "eos_token_id": self.bos_id,
            "name_or_path": "nanochat-rustbpe",
        }.items():
            try:
                setattr(self.tokenizer, name, value)
            except Exception:
                pass

    def _encode(self, text: str) -> List[int]:
        return [self.bos_id] + self.tokenizer.encode(text)

    @torch.inference_mode()
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        out: List[Tuple[float, bool]] = []
        for req in requests:
            context, continuation = req.args
            full_ids = self._encode(context + continuation)
            ctx_ids = self._encode(context)
            cont_start = min(len(ctx_ids), len(full_ids))

            drop = max(0, len(full_ids) - self.model_max_len)
            if drop > 0:
                full_ids = full_ids[drop:]
                cont_start = max(1, cont_start - drop)

            if cont_start >= len(full_ids):
                out.append((0.0, True))
                continue

            input_ids = torch.tensor(full_ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
            target_ids = torch.tensor(full_ids[1:], dtype=torch.long, device=self.device).unsqueeze(0)

            logits = self.model(input_ids)
            logprobs = torch.log_softmax(logits, dim=-1)

            start_pos = cont_start - 1
            cont_targets = target_ids[:, start_pos:]
            cont_logprobs = logprobs[:, start_pos:]
            token_lp = cont_logprobs.gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1)
            total_lp = float(token_lp.sum().item())

            greedy = cont_logprobs.argmax(dim=-1)
            is_greedy = bool((greedy == cont_targets).all().item())
            out.append((total_lp, is_greedy))
        return out

    @torch.inference_mode()
    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        out: List[Tuple[float]] = []
        for req in requests:
            (text,) = req.args
            ids = self._encode(text)
            if len(ids) <= 1:
                out.append((0.0,))
                continue
            drop = max(0, len(ids) - self.model_max_len)
            if drop > 0:
                ids = ids[drop:]
            input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
            target_ids = torch.tensor(ids[1:], dtype=torch.long, device=self.device).unsqueeze(0)
            logits = self.model(input_ids)
            logprobs = torch.log_softmax(logits, dim=-1)
            token_lp = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            out.append((float(token_lp.sum().item()),))
        return out

    @torch.inference_mode()
    def generate_until(self, requests) -> List[str]:
        outs: List[str] = []
        progress_enabled = os.environ.get("CETVEL_GENERATION_PROGRESS", "0") == "1"
        progress_every = max(1, int(os.environ.get("CETVEL_GENERATION_PROGRESS_EVERY", "1")))
        total = len(requests)
        started = time.monotonic()
        if progress_enabled:
            task_name = self._request_task_name(requests[0]) if total else "unknown"
            print0(f"[CETVEL generation] START task={task_name} total={total}", flush=True)

        for index, req in enumerate(requests, start=1):
            context, until = req.args
            until = until or []
            ids = self._encode(context)
            generated: List[int] = []
            stopped = False
            for token_column, _token_masks in self.engine.generate(
                ids,
                num_samples=1,
                max_tokens=self.max_gen_tokens,
                temperature=0.0,
            ):
                tok = int(token_column[0])
                if tok == self.bos_id:
                    stopped = True
                    break
                generated.append(tok)
                text = self.tokenizer.decode(generated)
                if any(stop in text for stop in until):
                    stopped = True
                    break
            outs.append(self.tokenizer.decode(generated))
            if progress_enabled and (index % progress_every == 0 or index == total):
                elapsed = max(1e-9, time.monotonic() - started)
                rate = index / elapsed
                task_name = self._request_task_name(req)
                print0(
                    "[CETVEL generation] "
                    f"task={task_name} done={index}/{total} "
                    f"generated_tokens={len(generated)} stopped={int(stopped)} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                    flush=True,
                )
        return outs

    @staticmethod
    def _request_task_name(req) -> str:
        task_name = getattr(req, "task_name", None)
        if task_name:
            return str(task_name)
        metadata = getattr(req, "metadata", None)
        if isinstance(metadata, (tuple, list)) and metadata:
            return str(metadata[0])
        return "unknown"
