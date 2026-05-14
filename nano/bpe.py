"""Byte-level BPE · 从零写一个能用的 tokenizer。

为什么 byte-level
=================
- 起手 256 个 byte 一定够用 · UTF-8 编码下任何字符都能拆成 1-4 个 byte
- 永远不会出现 ``<UNK>`` · 中文 / emoji / 古希腊语 / 私有 Unicode 区都能编
- GPT-2 / GPT-3 / GPT-4 / Llama-3 都是这套思路

算法 (Sennrich et al. 2016 · Karpathy minBPE)
============================================
1. 把语料编码成 byte 序列 · vocab 起手 = 256 个 byte
2. 统计相邻 pair 的频次
3. 把出现最多的 pair (a, b) 合成一个新 token id (= 256 + i)
4. 在语料里把所有 (a, b) 替换为新 id
5. 回 step 2 · 重复 max_merges 次

API
===
>>> from nano.bpe import BPE
>>> bpe = BPE.train("hello world hello", vocab_size=300)
>>> ids = bpe.encode("hello")
>>> bpe.decode(ids)
'hello'
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


def _get_pair_counts(ids: list[int]) -> Counter[tuple[int, int]]:
    """统计相邻 pair 的频次。``ids`` = 单段语料的 token id 序列。"""
    counts: Counter[tuple[int, int]] = Counter()
    for a, b in zip(ids, ids[1:], strict=False):
        counts[(a, b)] += 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """把 ids 里所有相邻的 ``pair`` 替换为 ``new_id``。

    例：ids=[1,2,3,1,2], pair=(1,2), new_id=99 → [99, 3, 99]
    """
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


@dataclass
class BPE:
    """Byte-level BPE · 训练 + 序列化 + 编解码 · 一个类全包。"""

    # merges[(a, b)] = new_id · 学习顺序 = id 升序
    merges: dict[tuple[int, int], int] = field(default_factory=dict)
    # vocab[id] = bytes · 反查每个 token 对应的 byte 串
    vocab: dict[int, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.vocab:
            # 256 个 byte 是天然的 base vocab
            self.vocab = {i: bytes([i]) for i in range(256)}
            for (a, b), nid in self.merges.items():
                self.vocab[nid] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ train
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        verbose: bool = False,
    ) -> BPE:
        """在语料 ``text`` 上学到 ``vocab_size`` 大的 BPE。

        必须 vocab_size ≥ 256 (要为 base byte 留位置)。
        每一轮挑频次最高的 pair 合 · 平局选 id 最小的（决定性 · 可复现）。
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size 必须 ≥ 256 · 给的 {vocab_size}")
        num_merges = vocab_size - 256

        # 把整段文本变成 byte id 序列 · 教学版不做 pre-tokenization
        # 工业版（industrial/）用 GPT-2 regex 切词 · 见那一节的注释
        ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            counts = _get_pair_counts(ids)
            if not counts:
                break  # 语料退化到长度 < 2 · 提前停
            # max 平局时选 id 最小 · 让结果对 random 稳定
            best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            pair, freq = best
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i+1:4d}/{num_merges} · "
                      f"pair=({pair[0]:3d},{pair[1]:3d}) "
                      f"freq={freq:6d} → id={new_id}")

        return cls(merges=merges, vocab=vocab)

    # ------------------------------------------------------------------ encode
    def encode(self, text: str) -> list[int]:
        """把字符串编成 token id 序列。

        实现策略：从 byte 序列开始 · 反复找<em>id 最小</em>的 merge 应用 · 直到没有可合的 pair。
        "id 最小" = "学得最早" = "频次最高的优先" · 跟训练顺序一致。
        """
        ids = list(text.encode("utf-8"))
        if len(ids) < 2:
            return ids
        while True:
            counts = _get_pair_counts(ids)
            # 在 ids 当前出现的 pair 里 · 找训练时学过且 id 最小的那个
            applicable = [(self.merges[p], p) for p in counts if p in self.merges]
            if not applicable:
                break
            _, pair = min(applicable, key=lambda x: x[0])
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    # ------------------------------------------------------------------ decode
    def decode(self, ids: list[int]) -> str:
        """token id 序列 → 字符串。

        非法 byte 序列（不可能 · 但保险）用 errors='replace' 渡过。
        """
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ I/O
    def save(self, path: Path | str) -> None:
        """把 merges 序列化成 JSON · vocab 用 hex 表示 byte 防止编码问题。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # merges 排序按 new_id · 反序列化时按这个顺序填回 self.merges
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        blob = {
            "version": "nano-bpe-v1",
            "merges": [{"a": a, "b": b, "id": nid} for (a, b), nid in ordered],
        }
        p.write_text(json.dumps(blob, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> BPE:
        blob = json.loads(Path(path).read_text())
        if blob.get("version") != "nano-bpe-v1":
            raise ValueError(f"未知 BPE 文件版本: {blob.get('version')!r}")
        merges = {(m["a"], m["b"]): m["id"] for m in blob["merges"]}
        return cls(merges=merges)
