"""
Estimate nanochat model sizes and Chinchilla-token horizons.

This is a dry-run planner: it does not load data or checkpoints.
"""

import argparse
import math


def ceil_to(x, m):
    return ((x + m - 1) // m) * m


def count_params(depth, vocab_size=32768, aspect_ratio=64, head_dim=128):
    model_dim = ceil_to(depth * aspect_ratio, head_dim)
    num_heads = model_dim // head_dim
    ve_layers = sum(1 for i in range(depth) if i % 2 == (depth - 1) % 2)
    wte = vocab_size * model_dim
    lm_head = vocab_size * model_dim
    value_embeds = ve_layers * vocab_size * model_dim
    transformer_matrices = depth * 12 * model_dim * model_dim
    transformer_matrices += ve_layers * 12 * num_heads  # tiny value-embedding gates
    scalars = 2 * depth + 24 + 1 + 1
    total = wte + lm_head + value_embeds + transformer_matrices + scalars
    scaling = transformer_matrices + lm_head
    return {
        "depth": depth,
        "model_dim": model_dim,
        "num_heads": num_heads,
        "wte": wte,
        "lm_head": lm_head,
        "value_embeds": value_embeds,
        "transformer_matrices": transformer_matrices,
        "scalars": scalars,
        "total": total,
        "scaling": scaling,
    }


def estimate_batch_size(target_tokens, d_ref, b_ref=2**19):
    predicted = b_ref * (target_tokens / d_ref) ** 0.383
    return 2 ** round(math.log2(predicted))


def fmt_billions(x):
    return f"{x / 1e9:.2f}B"


def fmt_millions(x):
    return f"{x / 1e6:.1f}M"


def main():
    parser = argparse.ArgumentParser(description="Estimate nanochat params@tokens")
    parser.add_argument("--depths", type=str, default="4,8,12,16,20,24,26")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--target-param-data-ratio", type=float, default=20)
    parser.add_argument("--target-param-count", type=str, default="total", choices=["total", "scaling"])
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    depths = [int(d.strip()) for d in args.depths.split(",") if d.strip()]
    d12 = count_params(12, args.vocab_size, args.aspect_ratio, args.head_dim)
    d_ref = args.target_param_data_ratio * d12[args.target_param_count]

    header = [
        "depth", "dim", "heads", "total_params", "scaling_params",
        "target_tokens", "auto_batch", "iters", "tokens/total", "tokens/scaling",
    ]
    print(",".join(header))
    for depth in depths:
        counts = count_params(depth, args.vocab_size, args.aspect_ratio, args.head_dim)
        target_tokens = int(args.target_param_data_ratio * counts[args.target_param_count])
        batch = estimate_batch_size(target_tokens, d_ref)
        iters = target_tokens // batch
        actual_tokens = iters * batch
        row = [
            str(depth),
            str(counts["model_dim"]),
            str(counts["num_heads"]),
            fmt_millions(counts["total"]),
            fmt_millions(counts["scaling"]),
            fmt_billions(actual_tokens),
            str(batch),
            str(iters),
            f"{actual_tokens / counts['total']:.2f}",
            f"{actual_tokens / counts['scaling']:.2f}",
        ]
        print(",".join(row))


if __name__ == "__main__":
    main()
