# CLAUDE.md · 给 AI coding agent 的项目上下文

> 让任意 AI coding agent（Claude Code / Cursor / Codex / Devin / 自家 agent）
> 在 30 秒内理解 taotao-llm 的全貌、约定、不该踩的坑。

---

## 1. 这是什么

教学 + 工业双轨的 LLM 训练 reference 项目。从 0 训一个 LLM 到上线：
tokenizer → pretrain → SFT → LoRA → DPO → eval → 量化 → 部署。

姊妹项目 `taotao-agent`（agent 全栈）是它的下游：训完模型推过去给 agent 用。

定位三句话：

1. **能跑的代码 + 配套讲解** —— 不是 paper 复读机
2. **macOS 友好** —— Mac (MPS) 跑 toy + LoRA · 大模型走云
3. **双轨**：`nano/` 教学版（≤300 行 / 文件，纯 PyTorch）和 `industrial/` 工业版（HF 全家桶）并行

---

## 2. 仓库地图（必读）

```
nano/         教学版 · 纯 PyTorch · 每个文件能独立读懂
industrial/   工业版 · transformers + trl + peft + accelerate + datasets
frontend/     React + Vite · 训练 dashboard + chat with my model
server/       FastAPI · 给前端用 · 暴露 /train/status · /chat · /eval
docs/         10 本 HTML 书（Phase 0-9 各一）· 跟 taotao-agent 同模板
data/         checkpoints + datasets · gitignored
notebooks/    探索性 jupyter
scripts/      one-shot 工具（download_dataset / convert_to_gguf / ...）
```

任何新增文件请先想清楚归哪一层 · 不要混着塞。

---

## 3. 学习路径 = 仓库 phase（10 个）

| Phase | 主题 | 关键代码 |
|---|---|---|
| 0 | 心智地图 | （只有书 · 无代码）|
| 1 | nanoGPT | `nano/gpt.py` + `nano/train.py` |
| 2 | Tokenizer | `nano/tokenize.py` + `industrial/tokenizer/` |
| 3 | Pretrain | `industrial/pretrain/` |
| 4 | SFT | `nano/sft.py` + `industrial/sft/` |
| 5 | LoRA / QLoRA | `industrial/lora/` |
| 6 | Preference (DPO) | `nano/dpo.py` + `industrial/dpo/` |
| 7 | Eval | `industrial/eval/` |
| 8 | Infra (FSDP / FlashAttention) | `industrial/infra/` |
| 9 | 量化 + 部署 | `industrial/deploy/` |

每完成一个 phase：
1. 代码进对应目录
2. `docs/0X-xxx.html` 写完（用 taotao-agent 的书模板）
3. README badge 推进 `phase-X/10`
4. commit message: `phase(X): <subject>`
5. tag `vphase-X`

---

## 4. 技术栈 · 锁定版本

```
Python 3.12
torch 2.4+ (mps for Mac)
transformers 4.45+
trl 0.11+
peft 0.13+
accelerate 1.0+
datasets 3.0+
tokenizers 0.20+
wandb (optional)
ruff + mypy + pytest
```

**不要引入：** TensorFlow、JAX、自训框架（除非 phase 8 实验对比）。
**可以加：** unsloth (phase 5 加速)、flash-attn (phase 8)、bitsandbytes (LoRA)。

frontend：
```
React 19 · Vite · TypeScript · @tanstack/react-query · recharts (loss curve)
```
不要引入 styled-components / Redux / Next.js。

---

## 5. 编码约定

### Python
- ruff format（line 100）+ ruff check（`E,F,I,UP,B,C4,SIM`）
- mypy 新代码必须 pass · 老代码 `# type: ignore[reason]` 标
- 函数优先有 docstring（Google style 1-2 句）
- Tensor shape 注释格式：`# (B, T, C)`（batch / time / channels）
- 常量 ALL_CAPS · 配置走 `dataclasses.dataclass`
- 不写 `from foo import *`

### nano/ 特殊约定
- 每个文件 ≤ 300 行
- 不引入除 `torch / numpy` 以外的依赖
- 有 `if __name__ == "__main__"` 块演示用法
- 注释多于代码 · 这是教学版

### industrial/ 特殊约定
- 走 HF 标准接口（Trainer / SFTTrainer / DPOTrainer）
- 配置走 yaml + Pydantic schema
- 一切实验 wandb 可选记录（`WANDB_DISABLED=true` 时静默）

---

## 6. Make 命令

```bash
make install        # uv sync 装全套
make smoke          # 30s import + MPS check
make nano-train     # phase 1 toy 模型在 tiny shakespeare 上 demo
make industrial-sft # phase 4 SFT 一个 small base model
make eval           # phase 7 lm-eval-harness 跑选定模型
make front-dev      # 前端 dev server
make server-dev     # FastAPI dashboard backend
make test           # pytest
make lint           # ruff + mypy
```

---

## 7. 数据 / 模型 / checkpoint 怎么放

- 数据：`data/datasets/<dataset_name>/`（HF datasets 缓存默认走这）
- checkpoint：`data/checkpoints/<run_name>/<step_xxxxxx>/`
- 量化产物：`data/gguf/<model_name>.<quant>.gguf`
- 全部 git-ignored · 一个新 contributor clone 后跑 `make download-tiny` 自动拉 demo 数据

环境变量约定（`.env.example` 给模板）：

```
HF_TOKEN=hf_xxx                    # huggingface 私有模型 / 推送
WANDB_API_KEY=xxx                  # 可选 · log to wandb
WANDB_PROJECT=taotao-llm           # 默认
WANDB_DISABLED=true                # 不想用就这个
HF_HOME=./data/hf_cache            # HF 缓存重定向到 repo 内
TORCH_DEVICE=mps                   # mps | cuda | cpu · 自动 detect 兜底
```

---

## 8. macOS / MPS 已知坑

- `torch.compile` 在 mps 上 2.4 还不稳 · 默认关
- `bitsandbytes` 不支持 mps · QLoRA 在 Mac 上跑不了 · 走纯 LoRA
- `flash-attn` 不支持 mps · 用 PyTorch SDPA 后端
- 内存监控：`torch.mps.current_allocated_memory()` · 不像 cuda 有 `nvidia-smi`
- 大于 14GB 的模型在 16GB Mac 上会 swap · 训练直接卡死 · 先 `du` 估
- 推理 7B+ 模型走 GGUF + llama.cpp · 别用 transformers

---

## 9. 跟 taotao-agent 的接合点

phase 9 deploy 完后约定：

- 模型推到 `~/.ollama/models/taotao-llm-{phase}-{quant}.gguf`
- 在 taotao-agent 的 `backend/.env` 写 `MODEL_NAME=taotao-llm-7b-q4`
- Ollama 服务 expose 给 agent · agent 不需要改代码

---

## 10. 别做的事

- ❌ 把 model checkpoint commit 进 git（用 HF Hub / wandb artifact）
- ❌ 在 nano/ 里 import transformers
- ❌ 把超过 100MB 的文件放进 repo
- ❌ 在 PR 里改超过 1 个 phase 的内容（碎片化 PR 更好）
- ❌ 在 .env 里塞真 token 后 commit（pre-commit gitleaks 会挡）
- ❌ 跟 transformers 升级不兼容时硬扛 · pin 老版本即可

---

## 11. 当前状态（2026-05）

- Phase 0 · bootstrap completed · 仓库骨架 + README + 路线图
- Phase 1-9 · pending · 按顺序铺
- 没有 production 部署 · 没有 CI 跑模型测试（cost 原因）
- 前端只有占位 README

下一个 PR 应该开 Phase 1 nanoGPT。
