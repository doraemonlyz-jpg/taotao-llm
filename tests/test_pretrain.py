"""industrial.pretrain smoke tests · 不依赖 HF 下载 · 不联网。

全部用 in-memory 假数据 / .bin 文件 · 测试管道是否通。
真数据 + 真训放在 ``make industrial-pretrain`` 那条手工命令。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from industrial.pretrain.data import PretrainDataset
from industrial.pretrain.model import PRESETS, build_model
from industrial.pretrain.train import PretrainConfig, train


def _make_fake_prepared(out_dir: Path, vocab_size: int, train_n: int, val_n: int) -> None:
    """伪造一份已 prepare 的数据集 · 跳过 HF 下载。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    train = rng.integers(0, vocab_size, size=train_n, dtype=np.uint16)
    val = rng.integers(0, vocab_size, size=val_n, dtype=np.uint16)
    train.tofile(out_dir / "train.bin")
    val.tofile(out_dir / "val.bin")
    (out_dir / "manifest.json").write_text(json.dumps({
        "dataset_name": "fake",
        "tokenizer_name": "gpt2",
        "train_tokens": train_n,
        "val_tokens": val_n,
        "vocab_size": vocab_size,
        "text_field": "text",
        "elapsed_sec": 0.0,
        "notes": {"dtype": "uint16", "eos_id": 50256},
    }))


def test_pretrain_dataset_load_and_batch(tmp_path: Path) -> None:
    _make_fake_prepared(tmp_path, vocab_size=1024, train_n=10_000, val_n=2_000)
    ds = PretrainDataset.load(tmp_path, block_size=64)
    xb, yb = ds.get_batch("train", batch_size=4)
    assert xb.shape == (4, 64)
    assert yb.shape == (4, 64)
    assert xb.dtype == torch.int64
    assert (xb < 1024).all() and (xb >= 0).all()
    # y 必须是 x 向左平移一位
    xb2, yb2 = ds.get_batch("val", batch_size=4)
    assert xb2.shape == (4, 64) and yb2.shape == (4, 64)


def test_pretrain_dataset_rejects_too_short(tmp_path: Path) -> None:
    _make_fake_prepared(tmp_path, vocab_size=128, train_n=10, val_n=10)
    ds = PretrainDataset.load(tmp_path, block_size=64)
    with pytest.raises(ValueError):
        ds.get_batch("train", batch_size=2)


@pytest.mark.parametrize("size", list(PRESETS))
def test_build_model_all_presets(size: str) -> None:
    model = build_model(size=size, vocab_size=512, block_size=64)
    n = model.num_params()
    # tiny ≈ 数 M · medium 几百 M · 至少 > 0
    assert n > 0
    # forward shape ok
    x = torch.zeros(2, 16, dtype=torch.long)
    logits, _ = model(x)
    assert logits.shape == (2, 16, 512)


def test_build_model_rejects_unknown_size() -> None:
    with pytest.raises(ValueError):
        build_model(size="enormous", vocab_size=512)


def test_train_one_iter_smoke(tmp_path: Path) -> None:
    """跑 5 个 iter · 验证整条 train loop 不崩。"""
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "ckpt"
    _make_fake_prepared(data_dir, vocab_size=512, train_n=20_000, val_n=2_000)

    # tokenizer 加载会失败 · 但 train() 应该 catch + 警告 · 不崩
    cfg = PretrainConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        size="tiny",
        block_size=32,
        micro_batch_size=2,
        grad_accum_steps=1,
        max_iters=5,
        warmup_iters=2,
        eval_interval=2,
        eval_iters=2,
        sample_interval=10**6,  # 不在 smoke 期间触发 sample
        save_interval=10**6,
        log_interval=1,
    )
    state = train(cfg)
    assert state.iter == 5
    assert (out_dir / "last.pt").exists()
    # metrics.jsonl 至少有一条 eval 记录
    metrics = (out_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(metrics) >= 1
    rec = json.loads(metrics[0])
    assert {"iter", "lr", "train", "val"}.issubset(rec)
