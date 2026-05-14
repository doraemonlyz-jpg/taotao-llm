"""tiny-shakespeare char-level 数据加载。

为什么 char-level
=================
- 词表只有 65 · 学起来直观 · 不需要 BPE 这种额外抽象
- 1MB 的语料 · 在 M 系列芯片上 5-10 分钟就能训出会写莎士比亚的小模型
- 把 tokenizer 留到 phase 2 单独学 · 这一阶段只关心 transformer
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_DATA_PATH = Path("data/datasets/tinyshakespeare/input.txt")


@dataclass
class CharDataset:
    """char-level 编码 + 90/10 train/val split。"""

    text: str
    block_size: int

    def __post_init__(self) -> None:
        chars = sorted(set(self.text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = dict(enumerate(chars))
        ids = np.array([self.stoi[c] for c in self.text], dtype=np.int64)
        n = int(len(ids) * 0.9)
        self.train_ids = ids[:n]
        self.val_ids = ids[n:]

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """随机抽 batch_size 条长度为 block_size 的窗口。

        target 就是 input 向左平移一位 · 经典语言建模目标。
        """
        ids = self.train_ids if split == "train" else self.val_ids
        # 随机起点 · 必须留出 block_size+1 的余量 (输入 + label)
        ix = np.random.randint(0, len(ids) - self.block_size - 1, size=(batch_size,))
        x = np.stack([ids[i : i + self.block_size] for i in ix])
        y = np.stack([ids[i + 1 : i + 1 + self.block_size] for i in ix])
        x_t = torch.from_numpy(x).to(device, non_blocking=True)
        y_t = torch.from_numpy(y).to(device, non_blocking=True)
        return x_t, y_t


def load_tinyshakespeare(
    block_size: int,
    path: Path | str = DEFAULT_DATA_PATH,
) -> CharDataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"找不到数据 {p} · 请先 `make download-tiny`"
        )
    text = p.read_text(encoding="utf-8")
    return CharDataset(text=text, block_size=block_size)
