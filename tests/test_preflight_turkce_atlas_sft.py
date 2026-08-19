import json

from scripts.preflight_turkce_atlas_sft import _verify_checkpoint


def test_verify_checkpoint_accepts_historical_nested_model_tag(tmp_path):
    step = 17100
    (tmp_path / f"model_{step:06d}.pt").write_bytes(b"model")
    for rank in range(4):
        (tmp_path / f"optim_{step:06d}_rank{rank}.pt").write_bytes(b"optimizer")
    (tmp_path / f"meta_{step:06d}.json").write_text(
        json.dumps(
            {
                "step": step,
                "model_config": {
                    "sequence_len": 2048,
                    "vocab_size": 32768,
                    "n_layer": 20,
                    "n_head": 10,
                    "n_kv_head": 10,
                    "n_embd": 1280,
                },
                "user_config": {"model_tag": "tr_d20_bpe_32768_chinchilla20"},
                "total_batch_size": 1048576,
            }
        ),
        encoding="utf-8",
    )

    identity = _verify_checkpoint(
        tmp_path,
        step,
        4,
        "tr_d20_bpe_32768_chinchilla20",
        1048576,
    )

    assert identity["model_tag"] == "tr_d20_bpe_32768_chinchilla20"
    assert identity["step"] == step
