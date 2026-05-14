"""nano-GPT · 教学版 decoder-only transformer。

设计原则
========
1. 每一行都能读懂 · shape 注释贴在右侧
2. 不依赖 flash-attn / xformers · 用 ``F.scaled_dot_product_attention`` 让 MPS 也能跑
3. 跟 Karpathy nanoGPT 接近 · 但去掉了 multi-platform 的开关 · 让初学者只关心 "transformer 长什么样"
4. config 用 ``dataclass`` 而不是 yaml · 一切在源码里看得到

用法
====
>>> import torch
>>> from nano.gpt import GPT, GPTConfig
>>> cfg = GPTConfig(vocab_size=65, block_size=64, n_layer=2, n_head=2, n_embd=64)
>>> model = GPT(cfg)
>>> x = torch.zeros(2, 16, dtype=torch.long)        # (B=2, T=16)
>>> logits, loss = model(x, targets=x)
>>> logits.shape, loss.item() > 0
(torch.Size([2, 16, 65]), True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """所有超参集中在这里 · 改一个数立刻生效。"""

    vocab_size: int = 65          # tiny-shakespeare char-level 65 个唯一字符
    block_size: int = 256         # context length T_max
    n_layer: int = 6              # transformer block 数
    n_head: int = 6               # multi-head attention head 数 · 必须能整除 n_embd
    n_embd: int = 384             # hidden size · 也叫 d_model
    dropout: float = 0.0          # 训练 tiny-shakespeare 不需要 dropout
    bias: bool = True             # Linear / LayerNorm 是否带 bias

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd 必须能整除 n_head"
        return self.n_embd // self.n_head


class CausalSelfAttention(nn.Module):
    """单层 multi-head causal self-attention。

    数学速记：
        Q = X · W_q,   K = X · W_k,   V = X · W_v        (B, T, C)
        att = softmax( (Q · Kᵀ) / √d_k  +  mask )        (B, h, T, T)
        out = att · V                                      (B, T, C)
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        # 一个 Linear 直接产出 q / k / v 三份 · 比开三个 Linear 快
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                              # (B, T, n_embd)
        qkv = self.c_attn(x)                           # (B, T, 3·C)
        q, k, v = qkv.split(C, dim=2)                  # 各 (B, T, C)
        # (B, T, C) → (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # PyTorch ≥ 2.0 内置 SDPA · is_causal=True 自动做下三角 mask
        # MPS 后端 0day 支持 SDPA · 不用我们手搓
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )                                              # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 拼回 (B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """两层 feed-forward · 中间放大 4× · GELU 激活 (原版 GPT 用的就是这套)。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)             # (B, T, 4C)
        x = F.gelu(x)                # 平滑版 ReLU
        x = self.c_proj(x)           # (B, T, C)
        return self.dropout(x)


class Block(nn.Module):
    """transformer block · pre-LayerNorm 结构 (后 GPT-2/3 主流写法)。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # 残差 + attention
        x = x + self.mlp(self.ln_2(x))    # 残差 + MLP
        return x


class GPT(nn.Module):
    """完整的 decoder-only transformer 语言模型。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)     # (vocab, C)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)     # (T_max, C)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying · GPT-2 的经典 trick · 输入 emb 和输出 head 共享权重
        # 直接减少 ≈ vocab × n_embd 的参数 · 同时实测 ppl 略好
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 论文 §2.3 · 残差里的 c_proj 用更小的初始化 · 防止深层方差爆炸
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """前向。

        Args:
            idx: (B, T) long · 输入 token id
            targets: (B, T) long or None · 给了就一起算 loss

        Returns:
            logits: (B, T, vocab_size)
            loss:   标量 cross-entropy 或 None
        """
        B, T = idx.shape
        max_T = self.cfg.block_size
        if T > max_T:  # noqa: SIM300 — shape convention: T is sequence length
            raise ValueError(f"序列长度 {T} > block_size {max_T}")

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)   # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)                       # (B, T, C)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)                                        # (B, T, vocab)
        loss = None
        if targets is not None:
            # cross_entropy 要求 (N, C) vs (N,) · flatten 一下
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """自回归采样。

        Args:
            idx:        (B, T0) 起手 prompt
            max_new_tokens: 要续写多少 token
            temperature: >0 · 越大越随机 · 1.0 = 原 distribution
            top_k:       None 表示 full distribution · 否则只在 top-k 里采样

        Returns:
            (B, T0 + max_new_tokens) 拼接好的 id 序列
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 截断到最后 block_size 个 token · 防止超长
            idx_cond = idx if idx.size(1) <= self.cfg.block_size \
                else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)          # (B, vocab)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)           # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
