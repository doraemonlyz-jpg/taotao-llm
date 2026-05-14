"""industrial.pretrain.train — 工业 pretrain 训练循环。

跟 ``nano.train`` 比 · 多了哪些工程能力
======================================
1. **梯度累积** (grad_accum_steps)：micro-batch 凑出更大的 effective batch · 不爆显存
2. **checkpoint resume**：重启训练继续 · iter / optim / scheduler 全恢复
3. **jsonl 结构化日志**：每 N 步写一行 · 给前端 dashboard / wandb 拽
4. **eval + sample 回调**：按 interval 算 val loss · 顺便生成几个 token 样本看质量
5. **可读 metrics**：tok/s · 已耗时 · ETA · loss · grad_norm

不做的事（留给 phase 8 infra）
==============================
- 多卡 / FSDP / DeepSpeed — 教学项目单卡足够
- mixed precision · MPS 下 fp16 不稳 · 默认 fp32
- FlashAttention — 已用 SDPA · 后端会自己选
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from industrial.pretrain.data import PretrainDataset
from industrial.pretrain.model import PRESETS, build_model
from nano.gpt import GPT
from nano.train import get_lr, pick_device


@dataclass
class PretrainConfig:
    # data + model
    data_dir: Path = Path("data/pretrain/tinystories")
    out_dir: Path = Path("data/ckpt/pretrain")
    size: str = "tiny"          # PRESETS key
    block_size: int = 512
    dropout: float = 0.0

    # batch
    micro_batch_size: int = 16   # 单次 forward 的 batch
    grad_accum_steps: int = 4    # effective batch = micro × accum
    grad_clip: float = 1.0

    # optim
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 200
    max_iters: int = 5000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # eval / log / save
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 25
    save_interval: int = 500
    sample_interval: int = 500
    sample_max_new: int = 80
    sample_prompt: str = "Once upon a time"

    # control
    seed: int = 1337
    resume: bool = False         # 若 out_dir/ckpt.pt 存在就接着训


@dataclass
class _RunState:
    iter: int = 0
    best_val: float = float("inf")
    history: list[dict] = field(default_factory=list)


def _save_ckpt(
    path: Path,
    model: GPT,
    optim: torch.optim.Optimizer,
    state: _RunState,
    cfg: PretrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "model_cfg": asdict(model.cfg),
        "train_cfg": _serialise_cfg(cfg),
        "iter": state.iter,
        "best_val": state.best_val,
    }, path)


def _serialise_cfg(cfg: PretrainConfig) -> dict:
    """dataclass → JSON-friendly dict (Path → str)。"""
    out = asdict(cfg)
    for k, v in out.items():
        if isinstance(v, Path):
            out[k] = str(v)
    return out


@torch.no_grad()
def _eval(
    model: GPT,
    ds: PretrainDataset,
    cfg: PretrainConfig,
    device: torch.device,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = ds.get_batch(split, cfg.micro_batch_size, device=device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


@torch.no_grad()
def _sample(
    model: GPT,
    tokenizer,
    prompt: str,
    max_new: int,
    device: torch.device,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new, temperature=0.8, top_k=40)
    return tokenizer.decode(out[0].tolist())


def train(cfg: PretrainConfig) -> _RunState:
    """主训练入口 · 返回最终 RunState。"""
    torch.manual_seed(cfg.seed)
    device = pick_device()
    print(f"[device] {device}")
    print(f"[size]   {cfg.size}  block_size={cfg.block_size}")

    ds = PretrainDataset.load(cfg.data_dir, block_size=cfg.block_size)
    print(f"[data]   {cfg.data_dir} · vocab={ds.vocab_size} · "
          f"train={len(ds.train_ids):,} · val={len(ds.val_ids):,} tokens")

    model = build_model(
        size=cfg.size, vocab_size=ds.vocab_size,
        block_size=cfg.block_size, dropout=cfg.dropout,
    ).to(device)
    n_params = model.num_params() / 1e6
    print(f"[model]  {n_params:.2f}M params · {PRESETS[cfg.size]}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    state = _RunState()
    ckpt = cfg.out_dir / "ckpt.pt"
    if cfg.resume and ckpt.exists():
        print(f"[resume] from {ckpt}")
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(blob["model_state"])
        optim.load_state_dict(blob["optim_state"])
        state.iter = blob["iter"]
        state.best_val = blob["best_val"]

    # 用于 eval 时 sample 看质量 · 跟 prepare 用的同款 tokenizer
    tokenizer = None
    manifest = json.loads((cfg.data_dir / "manifest.json").read_text())
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer_name"])
    except Exception as e:
        print(f"[warn] sample 不可用 (load tokenizer 失败): {e}")

    log_path = cfg.out_dir / "metrics.jsonl"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a")

    t0 = time.time()
    tokens_seen = 0
    try:
        while state.iter < cfg.max_iters:
            # nano.train.get_lr 只需要 4 个 attribute · PretrainConfig 同名 · 直接传
            lr = get_lr(state.iter, cfg)
            for g in optim.param_groups:
                g["lr"] = lr

            # ---- eval / sample / save ----
            if state.iter % cfg.eval_interval == 0 or state.iter == cfg.max_iters - 1:
                losses = _eval(model, ds, cfg, device)
                elapsed = time.time() - t0
                ts = tokens_seen / max(1.0, elapsed)
                rec = {
                    "iter": state.iter, "lr": lr,
                    "train": losses["train"], "val": losses["val"],
                    "tok_per_sec": ts, "elapsed_sec": elapsed,
                }
                state.history.append(rec)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(f"\n[iter {state.iter:5d}] lr={lr:.2e}  "
                      f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
                      f"tok/s={ts/1e3:.1f}k  ({elapsed:.0f}s)")

                if losses["val"] < state.best_val:
                    state.best_val = losses["val"]
                    _save_ckpt(ckpt, model, optim, state, cfg)
                    print(f"           ✓ saved best ckpt → {ckpt}")

            if (state.iter % cfg.sample_interval == 0
                    and tokenizer is not None and state.iter > 0):
                text = _sample(model, tokenizer,
                               cfg.sample_prompt, cfg.sample_max_new, device)
                print(f"           sample: {text!r}")

            # ---- gradient accumulation ----
            optim.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                xb, yb = ds.get_batch("train", cfg.micro_batch_size, device=device)
                _, loss = model(xb, yb)
                loss = loss / cfg.grad_accum_steps
                loss.backward()
                accum_loss += loss.item()
                tokens_seen += cfg.micro_batch_size * cfg.block_size

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            state.iter += 1

            if state.iter % cfg.log_interval == 0:
                ts = tokens_seen / max(1.0, time.time() - t0)
                print(f"  iter {state.iter:5d}  loss={accum_loss:.4f}  "
                      f"|grad|={float(grad_norm):.2f}  tok/s={ts/1e3:.1f}k", end="\r")
    finally:
        log_f.close()
        # 末尾再保一次 (即便不是 best · 也方便接着训)
        last = cfg.out_dir / "last.pt"
        _save_ckpt(last, model, optim, state, cfg)

    print(f"\n[done] iter={state.iter}  best_val={state.best_val:.4f}  "
          f"  ckpt={ckpt}  metrics={log_path}")
    return state
