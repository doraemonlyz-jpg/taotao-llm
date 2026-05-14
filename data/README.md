# data/

> 本目录被 `.gitignore` 全部 ignore（除了这个 README 和 `.gitkeep`）。

约定：

```
data/
├── datasets/        # HF datasets 缓存 + 自下载的语料
├── checkpoints/     # 训练 checkpoint · 一个 run 一个子目录
├── hf_cache/        # HF_HOME 重定向（环境变量在 .env 设）
├── gguf/            # phase 9 量化产物
└── eval/            # benchmark 结果 jsonl
```

`make download-tiny` 会把 phase 1 用的 ~1MB tiny shakespeare 拉到
`data/datasets/tinyshakespeare/input.txt`。
