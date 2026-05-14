# nano/ · 教学版

> 纯 PyTorch · 每个文件 ≤ 300 行 · 不引入除 `torch` / `numpy` 以外的依赖。
> 重在<strong>读懂每一行</strong>。

## 跟 industrial/ 的关系

| 同一概念 | nano/ | industrial/ |
|---|---|---|
| GPT 模型 | `gpt.py` (~150 行) | `transformers.AutoModelForCausalLM` |
| 训练 loop | `train.py` (~100 行) | `transformers.Trainer` |
| BPE tokenizer | `tokenize.py` (~200 行) | `tokenizers.ByteLevelBPETokenizer` |
| SFT | `sft.py` (~150 行) | `trl.SFTTrainer` |
| DPO | `dpo.py` (~180 行) | `trl.DPOTrainer` |

学习顺序：先读 `nano/` 知道每行在做什么 · 再换 `industrial/` 拿生产代码跑真东西。

## 当前状态

phase 0 · bootstrap · 文件还没写。phase 1 (nanoGPT) 是下一步。
