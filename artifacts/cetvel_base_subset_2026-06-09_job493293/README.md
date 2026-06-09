# CETVEL Base-Model Subset: Tasks 01-13

This folder contains the completed CETVEL results for the raw nanochat Turkish BPE base model before SFT.

- Model tag: `tr_d20_bpe_32768_chinchilla20`
- Model step: `17100`
- Checkpoint format: raw nanochat checkpoint, BPE tokenizer
- Training job id: `492421`
- CETVEL job id: `493293`
- W&B run: https://wandb.ai/nurcunal-bogaziciuniversitesi/nanochat-turk/runs/mygiras0

The benchmark was intentionally stopped after `tquad`. Tasks 01-13 cover the base-model selection and extractive-QA slice; the remaining suite entries are mostly open-ended generation/instruction-style tasks better reserved for the future SFT model.

Included suite entries: `exams_tr`, `belebele_tr`, `turkish_plu`, `cetvel_xcopa_tr`, `cetvel_xnli_tr`, `mnli_tr`, `snli_tr`, `news_cat`, `offenseval_tr`, `trclaim19`, `xfact_tr`, `xquad_tr`, `tquad`.

Deferred suite entries: `mkqa_tr`, WMT translation prompts, Turkish summarization tasks, and `gecturk_generation`.

Files:

- `cetvel_out_tasks_01_13.tar.gz`: complete compressed CETVEL output archive. It contains `cetvel_full_partial_results.json` and `task_results/` through `tquad`.
- `metrics_summary.json`: compact metric summary for quick inspection.
- `logs/`: SLURM stdout/stderr for the benchmark job.
- `manifest.json`: provenance and inclusion/exclusion rationale.

Expanded artifacts were also uploaded to Hugging Face:
https://huggingface.co/nurcunal/nanochat-turk-d20-bpe32k/tree/main/evaluation/cetvel_base_subset_tasks_01_13_job493293
