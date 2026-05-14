"""nano BPE smoke tests · 不依赖外部数据。"""

from __future__ import annotations

from pathlib import Path

import pytest

from nano.bpe import BPE

CORPUS_EN = (
    "the quick brown fox jumps over the lazy dog. "
    "the rain in spain falls mainly in the plain. "
    "she sells seashells by the seashore. "
)
CORPUS_MIX = "the cat sat on the mat. 测试中文。emoji 😀 也行。"


def test_base_vocab_is_256_bytes() -> None:
    bpe = BPE()
    assert bpe.vocab_size == 256
    for i in range(256):
        assert bpe.vocab[i] == bytes([i])


def test_train_grows_vocab_to_target() -> None:
    bpe = BPE.train(CORPUS_EN, vocab_size=300)
    assert bpe.vocab_size == 300
    assert len(bpe.merges) == 300 - 256


def test_encode_decode_roundtrip_english() -> None:
    bpe = BPE.train(CORPUS_EN, vocab_size=320)
    for s in ["the", "the quick", "fox jumps", "she sells seashells", ""]:
        assert bpe.decode(bpe.encode(s)) == s


def test_encode_decode_roundtrip_unicode() -> None:
    """byte-level BPE 必须能处理任何 UTF-8 · 即便训练时没见过。"""
    bpe = BPE.train(CORPUS_EN, vocab_size=300)  # 训练里全是 ASCII
    for s in ["中文测试", "café", "naïve", "résumé", "🎉🚀", "Mixed: hello 世界 🌍"]:
        assert bpe.decode(bpe.encode(s)) == s, f"roundtrip failed for {s!r}"


def test_compression_ratio_improves_with_vocab() -> None:
    """vocab 越大 · 平均 token 长度越长 · 压缩越好。"""
    text = CORPUS_EN * 10
    raw = len(text.encode("utf-8"))
    bpe_small = BPE.train(text, vocab_size=280)
    bpe_large = BPE.train(text, vocab_size=400)
    n_small = len(bpe_small.encode(text))
    n_large = len(bpe_large.encode(text))
    assert n_large < n_small, "更大词表应该 token 数更少"
    assert n_small <= raw, "BPE 不该比 byte-level 还差"


def test_save_and_load_preserve_behaviour(tmp_path: Path) -> None:
    bpe1 = BPE.train(CORPUS_EN, vocab_size=320)
    out = tmp_path / "bpe.json"
    bpe1.save(out)
    bpe2 = BPE.load(out)
    assert bpe2.vocab_size == bpe1.vocab_size
    assert bpe2.merges == bpe1.merges
    sample = "the quick brown fox 中文 🎉"
    assert bpe2.encode(sample) == bpe1.encode(sample)
    assert bpe2.decode(bpe2.encode(sample)) == sample


def test_train_rejects_too_small_vocab() -> None:
    with pytest.raises(ValueError):
        BPE.train("hello", vocab_size=200)


def test_encode_empty_string() -> None:
    bpe = BPE.train(CORPUS_EN, vocab_size=300)
    assert bpe.encode("") == []
    assert bpe.decode([]) == ""


def test_special_chars_no_unk() -> None:
    """byte-level 不应出现 UNK · 任何字节都能编。"""
    bpe = BPE.train(CORPUS_MIX, vocab_size=320)
    weird = "\x00\x01\x02 ZWJ‍ NBSP\xa0 emoji 👨‍💻 中"
    out = bpe.decode(bpe.encode(weird))
    assert out == weird


def test_deterministic_training() -> None:
    """同样语料 · 同样 vocab_size · 必须出同样的 merges (无随机性)。"""
    bpe1 = BPE.train(CORPUS_EN, vocab_size=320)
    bpe2 = BPE.train(CORPUS_EN, vocab_size=320)
    assert bpe1.merges == bpe2.merges
