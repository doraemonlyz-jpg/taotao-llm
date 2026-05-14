"""scaling-law mini 实验 · 用 3 档模型 × 3 个训练长度 = 9 个 run · 跑出 loss 表。

用法
====
make prepare-pretrain-data     # 先准备 10M token 的 TinyStories
uv run python scripts/scaling_law.py --quick   # 30 分钟跑完 (M2 air)
uv run python scripts/scaling_law.py --full    # 几小时

输出
====
- data/scaling/results.jsonl  · 每个 run 一行 · {size, iters, val_loss, params, ...}
- data/scaling/results.html   · 简单 HTML 表格 + 一段说明 · 浏览器直接看

物理意义
========
看 loss 怎么随<strong>参数量 N</strong>和<strong>训练 token 数 D</strong>下降。
你应该会看到 loss ≈ A + B·N^(-α) + C·D^(-β) 的趋势 · 这就是 Hoffmann/Chinchilla 经验。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from industrial.pretrain.train import PretrainConfig, train

GRID_QUICK = [
    # (size, iters)
    ("tiny", 200),
    ("tiny", 600),
    ("tiny", 1500),
    ("mini", 200),
    ("mini", 600),
    ("mini", 1500),
]

GRID_FULL = [
    ("tiny",  500), ("tiny",  1500), ("tiny",  3000),
    ("mini",  500), ("mini",  1500), ("mini",  3000),
    ("small", 500), ("small", 1500), ("small", 3000),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path,
                   default=Path("data/pretrain/tinystories"))
    p.add_argument("--out-dir", type=Path, default=Path("data/scaling"))
    p.add_argument("--quick", action="store_true",
                   help="用 6-run grid · M2 air 30 分钟")
    p.add_argument("--full", action="store_true",
                   help="用 9-run grid · 几小时")
    return p.parse_args()


def render_html(records: list[dict], out: Path) -> None:
    head = """<!doctype html><meta charset="utf-8">
<title>scaling-law results</title>
<style>
body{font-family:'Lora',serif;background:#0f1d35;color:#f5ecd2;padding:2rem;max-width:760px;margin:auto}
h1{font-family:'Fraunces',serif;color:#f4c95a}
table{width:100%;border-collapse:collapse;background:#1c3155;border-radius:8px;overflow:hidden}
th,td{padding:.6rem .9rem;text-align:left;border-bottom:1px solid #2c4570}
th{background:rgba(244,201,90,.1);text-transform:uppercase;font-size:.75rem;letter-spacing:.06em}
tr:last-child td{border-bottom:0}
.n{font-family:'JetBrains Mono',monospace;color:#8eb58f}
small{color:#7e8aa6}
</style>
<h1>scaling-law mini 实验结果</h1>
<small>你的 Mac 上跑出来的版本 · 跟 Chinchilla 论文趋势对得上 · 但绝对值不可比</small>
<table><thead><tr>
<th>size</th><th>params (M)</th><th>iters</th><th>tokens seen</th>
<th>val loss</th><th>train loss</th><th>elapsed (s)</th>
</tr></thead><tbody>"""
    rows = []
    for r in records:
        rows.append(
            f"<tr><td>{r['size']}</td>"
            f"<td class='n'>{r['params_M']:.1f}</td>"
            f"<td class='n'>{r['iters']}</td>"
            f"<td class='n'>{r['tokens_seen']:,}</td>"
            f"<td class='n'>{r['val_loss']:.4f}</td>"
            f"<td class='n'>{r['train_loss']:.4f}</td>"
            f"<td class='n'>{r['elapsed_sec']:.0f}</td></tr>"
        )
    out.write_text(head + "\n".join(rows) + "</tbody></table>")


def main() -> None:
    args = parse_args()
    grid = GRID_FULL if args.full else GRID_QUICK
    print(f"[grid] {len(grid)} runs · est ~{len(grid)*5} 分钟 (rough)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = args.out_dir / "results.jsonl"
    records: list[dict] = []

    for i, (size, iters) in enumerate(grid):
        print(f"\n══════ run {i+1}/{len(grid)}: size={size} iters={iters} ══════")
        ckpt = args.out_dir / f"{size}_{iters}"
        cfg = PretrainConfig(
            data_dir=args.data_dir,
            out_dir=ckpt,
            size=size,
            max_iters=iters,
            warmup_iters=min(50, iters // 10),
            eval_interval=max(50, iters // 4),
            eval_iters=20,
            log_interval=50,
            sample_interval=10**9,  # 关掉
        )
        state = train(cfg)
        last = state.history[-1] if state.history else {"train": -1, "val": -1}
        rec = {
            "size": size,
            "iters": iters,
            "params_M": _params_for(size, vocab_size=50257) / 1e6,
            "tokens_seen": cfg.micro_batch_size * cfg.grad_accum_steps
                * cfg.block_size * iters,
            "val_loss": last["val"],
            "train_loss": last["train"],
            "elapsed_sec": last.get("elapsed_sec", -1),
            "config": asdict(cfg) | {"data_dir": str(cfg.data_dir),
                                     "out_dir": str(cfg.out_dir)},
        }
        records.append(rec)
        with log.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    render_html(records, args.out_dir / "results.html")
    print(f"\n✓ {len(records)} runs · results: {log} · html: "
          f"{args.out_dir/'results.html'}")


def _params_for(size: str, vocab_size: int) -> int:
    """临时算个 size · 不真 build。"""
    from industrial.pretrain.model import PRESETS
    p = PRESETS[size]
    L, C = p.n_layer, p.n_embd
    return 12 * L * C * C + vocab_size * C  # weight tying 后


if __name__ == "__main__":
    main()
