# industrial/ · 工业版

> HuggingFace 全家桶 · 跟着社区主流姿势走。

## 子目录（按 phase 铺）

```
industrial/
├── tokenizer/   phase 2 · 用 tokenizers 库训 BPE / SentencePiece
├── pretrain/    phase 3 · transformers + accelerate · 分布式 ready
├── sft/         phase 4 · trl SFTTrainer · chat template + packing
├── lora/        phase 5 · peft LoRA / QLoRA
├── dpo/         phase 6 · trl DPOTrainer (also KTO / IPO)
├── eval/        phase 7 · lm-eval-harness wrapper + 自造 benchmark runner
├── infra/       phase 8 · FlashAttention / FSDP / DeepSpeed 配置示例
└── deploy/      phase 9 · convert-to-gguf / 推 vLLM / 注册到 Ollama
```

## 设计约定

- 每个子模块给 `run.py` (CLI 入口) + `config.yaml` (默认配置) + `README.md` (一页说明)
- 配置走 Pydantic dataclass · `python -m industrial.sft.run --config configs/qwen-0.5b.yaml`
- 一切实验 wandb 可选 · `WANDB_DISABLED=true` 时静默
- 不直接调 `transformers.Trainer.__init__` 时塞 50 个参数 · 走 helper builder

## 当前状态

phase 0 · bootstrap · 子目录还是空的 · phase 1 完成后立刻铺 phase 2 (tokenizer)。
