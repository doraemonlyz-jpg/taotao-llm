"""nano-GPT · char-level training loop。

为什么这么设计
==============
- 一个文件 ≈ 200 行 · 不抽 trainer / lightning / accelerate · 让你看到训练循环每一步
- ``argparse`` 暴露所有超参 · 跑 quick smoke 用 ``--max-iters 50``
- AdamW + cosine LR with warmup · 这套是 nanoGPT / Llama / Mistral 的标准 recipe
- 自动选 device · MPS > CUDA > CPU
- 训练完保存 ckpt 到 ``data/ckpt/nano-shakespeare.pt`` · sample.py 读它

跑通命令
========
make download-tiny          # 一次性
uv run python -m nano.train  # 默认 5000 步 · M2 air ~10 分钟
uv run python -m nano.train --max-iters 50 --eval-interval 25  # smoke
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nano.data import load_tinyshakespeare
from nano.gpt import GPT, GPTConfig

DEFAULT_CKPT = Path("data/ckpt/nano-shakespeare.pt")


@dataclass
class TrainConfig:
    block_size: int = 256
    batch_size: int = 32
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0

    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    max_iters: int = 5000
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 25
    seed: int = 1337


def pick_device() -> torch.device:
    """MPS > CUDA > CPU · 教学项目优先适配 M 系列。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr(it: int, cfg: TrainConfig) -> float:
    """linear warmup + cosine decay · 跟 GPT-3 / Llama 同款。"""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    dataset,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = dataset.get_batch(split, cfg.batch_size, device=device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    base = TrainConfig()
    for f, v in asdict(base).items():
        p.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    return TrainConfig(**{k.replace("-", "_"): v for k, v in vars(p.parse_args()).items()})


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    device = pick_device()
    print(f"[device] {device}")

    dataset = load_tinyshakespeare(block_size=cfg.block_size)
    print(f"[data]   vocab_size={dataset.vocab_size}  "
          f"train={len(dataset.train_ids):,}  val={len(dataset.val_ids):,}")

    model_cfg = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPT(model_cfg).to(device)
    print(f"[model]  params={model.num_params() / 1e6:.2f}M  "
          f"layers={cfg.n_layer}  heads={cfg.n_head}  d_model={cfg.n_embd}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    DEFAULT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best_val = float("inf")

    for it in range(cfg.max_iters):
        lr = get_lr(it, cfg)
        for g in optim.param_groups:
            g["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, dataset, cfg, device)
            elapsed = time.time() - t0
            print(f"[iter {it:5d}] lr={lr:.2e}  "
                  f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
                  f"({elapsed:.1f}s)")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model_state": model.state_dict(),
                    "model_cfg": asdict(model_cfg),
                    "train_cfg": asdict(cfg),
                    "stoi": dataset.stoi,
                    "itos": dataset.itos,
                    "iter": it,
                    "val_loss": best_val,
                }, DEFAULT_CKPT)
                print(f"           ✓ saved ckpt → {DEFAULT_CKPT}")

        xb, yb = dataset.get_batch("train", cfg.batch_size, device=device)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        if it % cfg.log_interval == 0:
            print(f"  iter {it:5d}  loss={loss.item():.4f}", end="\r")

    print(f"\n[done] best val={best_val:.4f}  ckpt={DEFAULT_CKPT}")


if __name__ == "__main__":
    main()
