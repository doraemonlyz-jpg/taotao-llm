"""nano-GPT smoke tests · 跑得快 · 不依赖数据。"""

from __future__ import annotations

import torch

from nano.gpt import GPT, GPTConfig


def _toy_cfg() -> GPTConfig:
    return GPTConfig(
        vocab_size=37,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )


def test_forward_shape() -> None:
    cfg = _toy_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 10))
    logits, loss = model(x)
    assert logits.shape == (3, 10, cfg.vocab_size)
    assert loss is None


def test_loss_is_finite_with_targets() -> None:
    cfg = _toy_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 8))
    _, loss = model(x, targets=x)
    assert loss is not None
    assert torch.isfinite(loss).all()
    # 随机初始化下 loss ≈ ln(vocab_size) · vocab=37 → ln(37)≈3.61
    # 给个宽容范围
    assert 1.0 < loss.item() < 6.0


def test_one_step_decreases_loss() -> None:
    """同一 batch 上做 30 步梯度下降 · loss 必须下降一大截。"""
    torch.manual_seed(0)
    cfg = _toy_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    _, loss0 = model(x, targets=x)
    for _ in range(30):
        _, loss = model(x, targets=x)
        optim.zero_grad()
        loss.backward()
        optim.step()
    _, loss_final = model(x, targets=x)

    assert loss_final.item() < loss0.item() * 0.5, (
        f"30 步还没把 loss 砍半 · 模型可能写错 · "
        f"loss0={loss0.item():.4f} → loss_final={loss_final.item():.4f}"
    )


def test_generate_extends_sequence() -> None:
    cfg = _toy_cfg()
    model = GPT(cfg)
    idx = torch.zeros(1, 3, dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (1, 8)
    assert (out >= 0).all() and (out < cfg.vocab_size).all()


def test_block_size_truncation() -> None:
    """generate 在长度超过 block_size 时应自动滑窗 · 不应崩。"""
    cfg = _toy_cfg()
    model = GPT(cfg)
    idx = torch.zeros(1, cfg.block_size, dtype=torch.long)  # 已经满了
    out = model.generate(idx, max_new_tokens=4, top_k=5)
    assert out.shape == (1, cfg.block_size + 4)


def test_weight_tying() -> None:
    cfg = _toy_cfg()
    model = GPT(cfg)
    assert model.tok_emb.weight.data_ptr() == model.lm_head.weight.data_ptr(), \
        "weight tying 没生效 · tok_emb 和 lm_head 应共享存储"
