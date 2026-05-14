"""加载训好的 nano-GPT ckpt · 续写一段。

用法
====
uv run python -m nano.sample                           # 默认起手 "ROMEO:"
uv run python -m nano.sample --prompt "JULIET:" --max-new-tokens 500 --temperature 0.8
uv run python -m nano.sample --top-k 40 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nano.gpt import GPT, GPTConfig
from nano.train import DEFAULT_CKPT, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--prompt", type=str, default="ROMEO:")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"[device] {device}")

    if not args.ckpt.exists():
        raise SystemExit(f"找不到 ckpt {args.ckpt} · 先 `make nano-train`")
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)

    model_cfg = GPTConfig(**blob["model_cfg"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    stoi: dict[str, int] = blob["stoi"]
    itos: dict[int, str] = blob["itos"]
    print(f"[model]  params={model.num_params() / 1e6:.2f}M  "
          f"trained_iter={blob.get('iter', '?')}  val_loss={blob.get('val_loss', '?'):.4f}")

    if any(c not in stoi for c in args.prompt):
        unknown = sorted({c for c in args.prompt if c not in stoi})
        raise SystemExit(f"prompt 含未在词表里的字符: {unknown!r}")
    idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = "".join(itos[int(i)] for i in out[0].tolist())
    print("─" * 60)
    print(text)
    print("─" * 60)


if __name__ == "__main__":
    main()
