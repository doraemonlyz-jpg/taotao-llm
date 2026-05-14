"""nano · BPE 训练入口。

用法
====
make download-tiny                       # 没数据先拉一下
uv run python -m nano.tokenize           # 默认在 shakespeare 上训 1024 vocab
uv run python -m nano.tokenize --vocab-size 2048 --out data/tokenizer/sp-2k.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from nano.bpe import BPE
from nano.data import DEFAULT_DATA_PATH

DEFAULT_OUT = Path("data/tokenizer/nano-bpe.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.corpus.exists():
        raise SystemExit(f"找不到语料 {args.corpus} · 先 `make download-tiny`")

    text = args.corpus.read_text(encoding="utf-8")
    raw_bytes = len(text.encode("utf-8"))
    print(f"[corpus]  {args.corpus} · {raw_bytes:,} bytes")
    print(f"[target]  vocab_size={args.vocab_size}  (256 base + {args.vocab_size - 256} merges)")

    t0 = time.time()
    bpe = BPE.train(text, vocab_size=args.vocab_size, verbose=not args.quiet)
    dt = time.time() - t0
    print(f"[done]    {dt:.1f}s · vocab_size={bpe.vocab_size}")

    bpe.save(args.out)
    print(f"[saved]   → {args.out}")

    sample = "ROMEO: But, soft! what light through yonder window breaks?"
    ids = bpe.encode(sample)
    snippet = text[: min(len(text), 50_000)]
    snippet_bytes = len(snippet.encode("utf-8"))
    snippet_ids = len(bpe.encode(snippet))
    cratio = snippet_bytes / max(1, snippet_ids)
    print(f"[demo]    encode({sample!r}) → {len(ids)} ids")
    print(f"          first 12 ids: {ids[:12]}")
    print(f"          decode roundtrip: {bpe.decode(ids)!r}")
    print(f"[compress] 1 token ≈ {cratio:.2f} bytes (越大压缩越好 · byte-level=1.0)")


if __name__ == "__main__":
    main()
