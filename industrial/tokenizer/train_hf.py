"""industrial · 用 HuggingFace ``tokenizers`` 训 byte-level BPE · GPT-2 同款配方。

vs nano 版的差异（这就是这个 phase 想让你看到的）
================================================
1. **预切词**：用 GPT-2 那条经典 regex 先把文本切成 word-ish 单元 ·
   防止 "the" 和 " the" 学成不同 token · 也防止 "a." 和 "a" 不一致
2. **Byte-level alphabet**：HF 用一个 256→可见字符 的双射映射 ·
   让 BPE 的中间产物全是可打印字符（方便序列化 / 调试）
3. **特殊 token**：BOS / EOS / PAD / UNK · 工业模型必备
4. **多线程**：HF 的 train_from_iterator 用 Rust 实现 · 比 nano 快 100x

用法
====
make download-tiny                       # 复用 phase 1 的 shakespeare
uv run python -m industrial.tokenizer.train_hf --vocab-size 4096
uv run python -m industrial.tokenizer.train_hf --corpus data/datasets/multilingual.txt --vocab-size 32000
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

DEFAULT_CORPUS = Path("data/datasets/tinyshakespeare/input.txt")
DEFAULT_OUT = Path("data/tokenizer/hf-bpe.json")

# GPT-2 那条出名 regex · 把数字 / 字母 / 符号 / 空白拆开
# 注意要前置一个空格再切 · 这样 " the" 和 "the" 都进同一个 pattern
GPT2_REGEX = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|"""
    r"""\p{N}{1,3}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|user|>", "<|assistant|>", "<|system|>"]


def _iter_lines(path: Path, batch: int = 4096) -> Iterator[list[str]]:
    """流式按行喂给 trainer · 大语料下避免一次性 read。"""
    chunk: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= batch:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def build_tokenizer() -> Tokenizer:
    """组装一个 byte-level BPE tokenizer · GPT-2 / GPT-3 同款流水线。"""
    tok = Tokenizer(models.BPE(unk_token=None))

    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=Regex(GPT2_REGEX),
            behavior="isolated",
            invert=False,
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-frequency", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.corpus.exists():
        raise SystemExit(f"找不到语料 {args.corpus} · 先 `make download-tiny`")

    raw_bytes = args.corpus.stat().st_size
    print(f"[corpus]  {args.corpus} · {raw_bytes:,} bytes")
    print(f"[target]  vocab_size={args.vocab_size}  specials={SPECIAL_TOKENS}")

    tok = build_tokenizer()
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=initial_alphabet,
        show_progress=True,
    )

    t0 = time.time()
    tok.train_from_iterator(
        _iter_lines(args.corpus),
        trainer=trainer,
    )
    dt = time.time() - t0
    print(f"[done]    {dt:.1f}s · vocab_size={tok.get_vocab_size()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(args.out))
    print(f"[saved]   → {args.out}")

    sample = "ROMEO: But, soft! what light through yonder window breaks?"
    enc = tok.encode(sample)
    print(f"[demo]    {sample!r}")
    print(f"          {len(enc.ids)} tokens")
    print(f"          ids   = {enc.ids[:14]}...")
    print(f"          tokens = {enc.tokens[:14]}...")
    print(f"          decode = {tok.decode(enc.ids)!r}")


if __name__ == "__main__":
    main()
