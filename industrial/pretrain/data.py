"""pretrain 数据流水线 · HF 数据集 → tokenize → pack → .bin 缓存。

整体策略 (Karpathy nanoGPT / TinyStories 同款)
=============================================
1. **流式拉数据**：用 HuggingFace ``datasets`` 的 ``streaming=True`` · 不一口气下载全集
2. **tokenize**：每条 example 过 tokenizer · 末尾加 EOS · 拼成 long stream
3. **pack**：把 stream 切成等长窗口（训练时再随机抽 batch_size 条）
4. **缓存到 .bin**：uint16 (vocab ≤ 65535) 或 uint32 · memmap 友好 · 训练时零拷贝

为什么要 .bin 缓存 · 不每次现 tokenize
=====================================
- HF tokenizer 即便快 · 1B token 也要数小时
- pretrain 通常多 epoch 跑同一份数据 · 重复 tokenize 浪费
- .bin 直接 memmap · 训练 dataloader 几乎零开销

API
===
>>> from industrial.pretrain.data import PretrainDataset
>>> PretrainDataset.prepare(dataset_name="roneneldan/TinyStories",
...                         tokenizer_name="gpt2",
...                         out_dir="data/pretrain/tinystories",
...                         max_train_tokens=10_000_000)  # 10M tokens
>>> ds = PretrainDataset.load("data/pretrain/tinystories", block_size=512)
>>> xb, yb = ds.get_batch("train", batch_size=8)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class PrepareStats:
    """记录 prepare() 跑出来的元数据 · 落盘成 manifest.json。"""

    dataset_name: str
    tokenizer_name: str
    train_tokens: int
    val_tokens: int
    vocab_size: int
    text_field: str
    elapsed_sec: float
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class PretrainDataset:
    """已经 tokenize+pack 好的数据集 · memmap 加载。"""

    train_ids: np.ndarray  # memmap · uint16 / uint32
    val_ids: np.ndarray
    block_size: int
    vocab_size: int

    # ----------------------------------------------------------- prepare
    @staticmethod
    def prepare(
        dataset_name: str,
        out_dir: Path | str,
        tokenizer_name: str = "gpt2",
        text_field: str = "text",
        split_train: str = "train",
        split_val: str = "validation",
        max_train_tokens: int | None = None,
        max_val_tokens: int | None = 1_000_000,
        verbose: bool = True,
    ) -> PrepareStats:
        """从 HF Hub 拉一个数据集 · tokenize · 写到 ``out_dir/{train,val}.bin``。

        Args:
            dataset_name: HF Hub 上的数据集名 · e.g. "roneneldan/TinyStories"
            out_dir:      .bin 文件输出目录 · 同时写 manifest.json
            tokenizer_name: HF tokenizer · 默认 "gpt2" (50k vocab · 中文不友好但稳)
            text_field:   每条 example 取哪个字段当文本 · 默认 "text"
            split_train / split_val: HF dataset 的 split 名
            max_train_tokens: 累计够这么多 token 就停 · None = 全集
            max_val_tokens:   val 同理 · 默认 1M token (~1MB)
            verbose: 进度打印
        """
        from datasets import load_dataset
        from transformers import AutoTokenizer

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"[prepare] dataset={dataset_name}  tokenizer={tokenizer_name}")

        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tok.eos_token_id is None:
            # 部分 tokenizer 没 EOS · 加一个 (例：bert)
            tok.add_special_tokens({"eos_token": "<|endoftext|>"})
        eos_id = tok.eos_token_id
        vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok)
        # 实际算 dtype · uint16 上限 65535 · 大 vocab 必须 uint32
        dtype = np.uint16 if vocab_size < 2**16 else np.uint32
        if verbose:
            print(f"[prepare] vocab_size={vocab_size}  dtype={dtype.__name__}  eos_id={eos_id}")

        t0 = time.time()
        train_n = _tokenize_and_pack(
            load_dataset(dataset_name, split=split_train, streaming=True),
            tok=tok, eos_id=eos_id, text_field=text_field,
            out_path=out / "train.bin", max_tokens=max_train_tokens,
            dtype=dtype, label="train", verbose=verbose,
        )
        val_n = _tokenize_and_pack(
            load_dataset(dataset_name, split=split_val, streaming=True),
            tok=tok, eos_id=eos_id, text_field=text_field,
            out_path=out / "val.bin", max_tokens=max_val_tokens,
            dtype=dtype, label="val", verbose=verbose,
        )
        elapsed = time.time() - t0

        stats = PrepareStats(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            train_tokens=train_n,
            val_tokens=val_n,
            vocab_size=vocab_size,
            text_field=text_field,
            elapsed_sec=elapsed,
            notes={"dtype": dtype.__name__, "eos_id": eos_id},
        )
        (out / "manifest.json").write_text(json.dumps(stats.__dict__, indent=2))
        if verbose:
            print(f"[done] train={train_n:,}  val={val_n:,}  in {elapsed:.1f}s "
                  f"→ {out}")
        return stats

    # ----------------------------------------------------------- load
    @classmethod
    def load(cls, prepared_dir: Path | str, block_size: int) -> PretrainDataset:
        d = Path(prepared_dir)
        manifest = json.loads((d / "manifest.json").read_text())
        dtype = np.uint16 if manifest["notes"]["dtype"] == "uint16" else np.uint32
        train = np.memmap(d / "train.bin", dtype=dtype, mode="r")
        val = np.memmap(d / "val.bin", dtype=dtype, mode="r")
        return cls(
            train_ids=train,
            val_ids=val,
            block_size=block_size,
            vocab_size=manifest["vocab_size"],
        )

    # ----------------------------------------------------------- batch
    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.train_ids if split == "train" else self.val_ids
        # 至少要 block_size+1 长 · 防越界
        max_start = len(ids) - self.block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"{split} 数据太短: {len(ids)} ≤ block_size={self.block_size}"
            )
        ix = np.random.randint(0, max_start, size=(batch_size,))
        # memmap 切片返回 ndarray · 拷成 int64 给 PyTorch
        x = np.stack([ids[i : i + self.block_size].astype(np.int64) for i in ix])
        y = np.stack([ids[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix])
        x_t = torch.from_numpy(x).to(device, non_blocking=True)
        y_t = torch.from_numpy(y).to(device, non_blocking=True)
        return x_t, y_t


def _tokenize_and_pack(
    stream,
    *,
    tok,
    eos_id: int,
    text_field: str,
    out_path: Path,
    max_tokens: int | None,
    dtype: np.dtype,
    label: str,
    verbose: bool,
) -> int:
    """流式 tokenize 一个 split · 写到 out_path · 返回总 token 数。"""
    BUF_BYTES = 64 * 1024 * 1024  # 64MB 写缓冲
    buf: list[int] = []
    written = 0
    n_examples = 0
    t0 = time.time()

    with out_path.open("wb") as f:
        for ex in stream:
            text = ex.get(text_field)
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False)
            buf.extend(ids)
            buf.append(eos_id)  # 文档间分隔
            n_examples += 1

            if len(buf) * (2 if dtype == np.uint16 else 4) >= BUF_BYTES:
                arr = np.asarray(buf, dtype=dtype)
                arr.tofile(f)
                written += len(arr)
                buf.clear()
                if verbose:
                    rate = written / max(1e-6, time.time() - t0)
                    print(f"  [{label}] {written:>12,} tokens · "
                          f"{n_examples:>8,} docs · {rate/1e3:.1f} kt/s",
                          end="\r")
            if max_tokens is not None and written + len(buf) >= max_tokens:
                break

        if buf:
            arr = np.asarray(buf, dtype=dtype)
            arr.tofile(f)
            written += len(arr)

    if verbose:
        print(f"  [{label}] {written:>12,} tokens · {n_examples:>8,} docs"
              f" · done in {time.time()-t0:.1f}s")
    return written
