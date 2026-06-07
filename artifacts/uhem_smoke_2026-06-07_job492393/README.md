# UHeM smoke test artifact: job 492393

This folder captures the completed Altay/UHeM smoke test for the Turkish
nanochat base-model pipeline.

## Slurm

- Cluster: Altay
- Project/account: `nakane`
- Partition: `gpu2dq`
- Job ID: `492393`
- Node: `a119`
- State: `COMPLETED`
- Exit code: `0:0`
- Wall clock: `00:17:38`
- Requested resources: `1` node, `64` CPU cores, `1x NVIDIA A100 80GB`, `64G` RAM
- Memory utilized: `19.79 GB`

## Pipeline

- Script: `runs/uhem_smoke_nakane_a100.sbatch`
- Model tag: `tr_d2_bpe_32768_uhem_smoke_chinchilla20`
- Base directory on UHeM: `/ari/users/nunal/nanochat-turk-smoke-cache`
- Dataset: `HuggingFaceFW/fineweb-2`, config `tur_Latn`
- Training shards: `2` train shards plus final validation shard
- Vocabulary size: `32768`
- Tokenizer: `bpe_32768_uhem_smoke`
- Depth: `2`
- Total parameters: `12,976,182`
- Training tokens: `259,522,560`
- Tokens/total-params ratio: `19.9999`

## Metrics

- Minimum validation BPB: `1.239557`
- Final validation BPB: `1.239557`
- Base eval train BPB: `1.295173`
- Base eval val BPB: `1.239951`
- Training time: `11.60m`
- Peak GPU memory reported by training loop: `1420.94 MiB`
- Total training FLOPs: `7.347524e15`
- MFU: `2.51%`

The generated samples are intentionally low quality because this is a tiny
depth-2 infrastructure smoke test, not a quality run.

## Files

- `report.md`: generated nanochat report
- `nanochat-tr-smoke-a100-492393.out`: Slurm stdout, training/eval logs, and job statistics
- `nanochat-tr-smoke-a100-492393.err`: environment install, tokenizer, and checkpoint stderr logs
- `metrics.json`: compact machine-readable run summary
- `checkpoint/model_001980.pt`: model checkpoint at step 1980
- `checkpoint/meta_001980.json`: checkpoint metadata
- `checkpoint/optim_001980_rank0.pt`: optimizer state for rank 0
