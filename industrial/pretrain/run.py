"""industrial.pretrain.run — 一个 CLI 三个动作。

prepare : 拉数据 + tokenize + 缓存到 .bin
train   : 真训
sample  : 拿训好的 ckpt 续写一段文本

例
==
# 1. 准备数据 (10M token TinyStories · ~3-5 分钟 · ~20MB 缓存)
uv run python -m industrial.pretrain.run prepare \
    --dataset-name roneneldan/TinyStories --max-train-tokens 10000000

# 2. 训 tiny GPT (~10M params · M2 air ~30 分钟)
uv run python -m industrial.pretrain.run train --size tiny --max-iters 3000

# 3. 看模型现在会写啥
uv run python -m industrial.pretrain.run sample --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from industrial.pretrain.data import PretrainDataset
from industrial.pretrain.model import build_model
from industrial.pretrain.train import PretrainConfig
from industrial.pretrain.train import train as run_train
from nano.train import pick_device


def cmd_prepare(args: argparse.Namespace) -> None:
    PretrainDataset.prepare(
        dataset_name=args.dataset_name,
        out_dir=args.out_dir,
        tokenizer_name=args.tokenizer,
        text_field=args.text_field,
        split_train=args.split_train,
        split_val=args.split_val,
        max_train_tokens=args.max_train_tokens,
        max_val_tokens=args.max_val_tokens,
    )


def cmd_train(args: argparse.Namespace) -> None:
    cfg = PretrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        size=args.size,
        block_size=args.block_size,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        warmup_iters=args.warmup_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        sample_interval=args.sample_interval,
        sample_prompt=args.sample_prompt,
        resume=args.resume,
        seed=args.seed,
    )
    run_train(cfg)


def cmd_sample(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    device = pick_device()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"找不到 ckpt {ckpt_path} · 先 train")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_cfg = blob["train_cfg"]
    model_cfg = blob["model_cfg"]
    print(f"[ckpt] iter={blob['iter']} val={blob['best_val']:.4f} "
          f"size={train_cfg.get('size','?')}")

    model = build_model(
        size=train_cfg["size"],
        vocab_size=model_cfg["vocab_size"],
        block_size=model_cfg["block_size"],
    ).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    import json
    manifest = json.loads((Path(train_cfg["data_dir"]) / "manifest.json").read_text())
    tok = AutoTokenizer.from_pretrained(manifest["tokenizer_name"])

    ids = tok.encode(args.prompt, add_special_tokens=False)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k,
    )
    text = tok.decode(out[0].tolist())
    print("─" * 60)
    print(text)
    print("─" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pretrain")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="拉数据 + tokenize + 缓存")
    pp.add_argument("--dataset-name", default="roneneldan/TinyStories")
    pp.add_argument("--tokenizer", default="gpt2")
    pp.add_argument("--out-dir", type=Path,
                    default=Path("data/pretrain/tinystories"))
    pp.add_argument("--text-field", default="text")
    pp.add_argument("--split-train", default="train")
    pp.add_argument("--split-val", default="validation")
    pp.add_argument("--max-train-tokens", type=int, default=10_000_000)
    pp.add_argument("--max-val-tokens", type=int, default=1_000_000)
    pp.set_defaults(func=cmd_prepare)

    pt = sub.add_parser("train", help="训")
    pt.add_argument("--data-dir", type=Path,
                    default=Path("data/pretrain/tinystories"))
    pt.add_argument("--out-dir", type=Path,
                    default=Path("data/ckpt/pretrain"))
    pt.add_argument("--size", default="tiny",
                    choices=["tiny", "mini", "small", "medium"])
    pt.add_argument("--block-size", type=int, default=512)
    pt.add_argument("--micro-batch-size", type=int, default=16)
    pt.add_argument("--grad-accum-steps", type=int, default=4)
    pt.add_argument("--max-iters", type=int, default=3000)
    pt.add_argument("--learning-rate", type=float, default=6e-4)
    pt.add_argument("--warmup-iters", type=int, default=200)
    pt.add_argument("--eval-interval", type=int, default=250)
    pt.add_argument("--eval-iters", type=int, default=50)
    pt.add_argument("--sample-interval", type=int, default=500)
    pt.add_argument("--sample-prompt", default="Once upon a time")
    pt.add_argument("--resume", action="store_true")
    pt.add_argument("--seed", type=int, default=1337)
    pt.set_defaults(func=cmd_train)

    ps = sub.add_parser("sample", help="加载 ckpt 续写")
    ps.add_argument("--ckpt", type=Path,
                    default=Path("data/ckpt/pretrain/ckpt.pt"))
    ps.add_argument("--prompt", default="Once upon a time")
    ps.add_argument("--max-new-tokens", type=int, default=200)
    ps.add_argument("--temperature", type=float, default=0.8)
    ps.add_argument("--top-k", type=int, default=40)
    ps.set_defaults(func=cmd_sample)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
