"""industrial.pretrain.model — 复用 nano.gpt · 提供 pretrain 用的几档 preset。

设计意图
========
我们刻意<em>不</em>在 pretrain 阶段引入 RoPE / RMSNorm / SwiGLU 这些升级 ·
phase 8 (infra) 才统一处理。这里只做：
- 把 nano.gpt.GPTConfig 包装成 "size 名 → preset" 的工厂
- 计算非 embedding 参数量 · 让你能跟 Chinchilla 经验对得上号
"""

from __future__ import annotations

from dataclasses import dataclass

from nano.gpt import GPT, GPTConfig


@dataclass(frozen=True)
class Preset:
    """一组对应已知 size 名的 GPTConfig 默认值。"""

    n_layer: int
    n_head: int
    n_embd: int
    block_size: int = 512


# 命名跟 GPT-2 family 对齐 · 但在 vocab=50257 下参数量大概是：
#   tiny  ≈  10M  · M2 air pretrain ~30 分钟体感能跑
#   mini  ≈  30M  · 上半天
#   small ≈ 124M  · 单卡 GPU 一两天 (MPS 上太慢 · 仅作展示)
PRESETS: dict[str, Preset] = {
    "tiny":  Preset(n_layer=4,  n_head=4,  n_embd=256,  block_size=512),
    "mini":  Preset(n_layer=6,  n_head=8,  n_embd=512,  block_size=512),
    "small": Preset(n_layer=12, n_head=12, n_embd=768,  block_size=1024),  # GPT-2 small 同款
    "medium":Preset(n_layer=24, n_head=16, n_embd=1024, block_size=1024),  # GPT-2 medium
}


def build_model(
    size: str,
    vocab_size: int,
    block_size: int | None = None,
    dropout: float = 0.0,
) -> GPT:
    """按 ``size`` 名构出一个 GPT。"""
    if size not in PRESETS:
        raise ValueError(f"unknown size {size!r}; choose from {list(PRESETS)}")
    p = PRESETS[size]
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size or p.block_size,
        n_layer=p.n_layer,
        n_head=p.n_head,
        n_embd=p.n_embd,
        dropout=dropout,
    )
    return GPT(cfg)
