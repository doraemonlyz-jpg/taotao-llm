# taotao-llm

> 从 0 训出一个 LLM · 双轨实现（教学 nanoGPT · 工业 HuggingFace 栈）+ 配套 10 本书 + 监控/对话 UI · macOS MPS 友好。

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HF transformers](https://img.shields.io/badge/HF-transformers-yellow.svg)](https://github.com/huggingface/transformers)
[![Phase](https://img.shields.io/badge/phase-0%2F10-orange.svg)](#-学习路径)
[![status](https://img.shields.io/badge/status-bootstrap-tomato)]()

> 姊妹项目：[`taotao-agent`](https://github.com/doraemonlyz-jpg/taotao-agent) — 27 本书 · agent 全栈
> 这个项目是它的<strong>上游</strong>：先训出模型，再喂给 agent。

---

## 🎯 它解决什么

> "transformer 论文我读懂了 · pretrain / SFT / DPO / RLHF 概念也知道 · 可是 *从 0 训一个能用的 LLM* 我没动手过 · 不知道 1B 模型在 Mac 上能不能跑通 · 不知道 chat template 长啥样 · 不知道 LoRA 该用啥参数 · 不知道训完怎么 eval ..."

如果你点头超过 3 次 · 这个仓库就是给你写的。

我们用<strong>双轨实现</strong>把每个概念吃透：

- **`nano/`** · ~200 行 PyTorch 手搓 · CPU/MPS 能跑 · 重在<em>看懂每行</em>
- **`industrial/`** · transformers + trl + peft + accelerate · 工业事实标准 · 重在<em>真能用</em>

每个 phase 都会同时出现两份实现 · 一边读教学版理解原理 · 一边对照工业版学正确姿势。

---

## 📚 学习路径 · 10 phase · 10 本书

每 phase 一本配套 HTML 书 + 一个里程碑 commit。

| Phase | 主题 | 时长 | 你将能做到 | 书 |
|---|---|---|---|---|
| **0** | 心智地图 · attention / token / loss / autoregressive 直觉 | 1 周 | 看懂任何 LLM 论文 abstract | `docs/00-mental-model.html` *(planned)* |
| **1** ✅ | nanoGPT 手搓 · tiny shakespeare · CPU/MPS 跑通 | 2 周 | 自己写出能"说人话"的 toy GPT | [`docs/01-nanogpt.html`](./docs/01-nanogpt.html) |
| **2** | Tokenizer · char → BPE → SentencePiece · 自训 vocab | 1 周 | 知道为啥中文 tokenizer 那么坑 | `docs/02-tokenizer.html` |
| **3** | Pretrain · FineWeb 子集 · 100M-1B params · scaling law · LR schedule | 2-3 周 | 把 nanoGPT 升级成"能用"的 base 模型 | `docs/03-pretrain.html` |
| **4** | SFT · chat template · packing · masking · Alpaca/Dolly | 2 周 | base → instruction follow | `docs/04-sft.html` |
| **5** | LoRA / QLoRA · peft · 单卡微调 7B | 1 周 | 一张消费级卡 fine-tune Llama / Qwen | `docs/05-lora.html` |
| **6** | Preference · DPO / KTO / GRPO · trl | 2 周 | 让模型"听话 + 有偏好" | `docs/06-preference.html` |
| **7** | Eval · lm-eval-harness · MMLU / HumanEval / GSM8K · 自造 benchmark | 1-2 周 | 客观证明你的模型比 baseline 强 | `docs/07-eval.html` |
| **8** | Infra · FlashAttention · Liger · FSDP · DeepSpeed · mixed precision | 2-3 周 | 多卡 / 把单卡极限榨出来 | `docs/08-infra.html` |
| **9** | 量化 + 部署 · AWQ / GPTQ / GGUF · vLLM / Ollama / llama.cpp | 1 周 | 训好的模型推到 chat UI / API | `docs/09-deploy.html` |

> 总计 **3-4 个月全职** · 学完你就有一个 senior LLM engineer 的核心 muscle memory。
> 当前 phase: **1 · nanoGPT shipped** ✅ · `make download-tiny && make nano-train` 一条命令就能跑 · 配套 [Book 01](./docs/01-nanogpt.html) 已发车。

---

## 🏗 仓库结构

```
taotao-llm/
├── nano/              # 教学版 · 纯 PyTorch · 每个文件 < 300 行
│   ├── gpt.py         # phase 1: model
│   ├── train.py       # phase 1: training loop
│   ├── tokenize.py    # phase 2: tokenizer from scratch
│   ├── sft.py         # phase 4: minimal SFT
│   └── dpo.py         # phase 6: minimal DPO
├── industrial/        # 工业版 · transformers + trl + peft
│   ├── pretrain/      # phase 3
│   ├── sft/           # phase 4
│   ├── lora/          # phase 5
│   ├── dpo/           # phase 6
│   ├── eval/          # phase 7
│   └── deploy/        # phase 9
├── frontend/          # 监控 dashboard + chat with my model · React + Vite
│   ├── src/
│   │   ├── pages/
│   │   │   ├── DashboardPage.tsx   # loss curve · LR · throughput
│   │   │   ├── ChatPage.tsx        # 跟自己训的模型对话
│   │   │   └── EvalPage.tsx        # benchmark 结果可视化
│   │   └── api.ts
│   └── vite.config.ts
├── docs/              # 10 本书 + 学习路径 landing
├── data/              # checkpoints / datasets · gitignored
├── notebooks/         # 探索性 jupyter
├── scripts/           # one-shot 脚本（download / convert / etc.）
├── server/            # FastAPI · 给前端 dashboard + chat 用 · 后续加
├── pyproject.toml
├── Makefile
├── CLAUDE.md          # 给 AI coding agent 的项目上下文
└── AGENTS.md          # 给运行时 agent 的行为约定
```

---

## 🚀 开始

```bash
git clone git@github.com:doraemonlyz-jpg/taotao-llm.git
cd taotao-llm
make install        # uv sync 装 PyTorch + transformers 全家桶
cp .env.example .env
# 填 HF_TOKEN（去 huggingface.co/settings/tokens 拿）
make smoke          # 跑一个 30 秒的 import + MPS check · 确认环境 OK
```

Mac 用户必看：

- M1/M2/M3 默认走 MPS（PyTorch 的 Apple GPU 后端）· 比 CPU 快 5-20×
- toy 模型（&lt; 100M params）能在 Mac 上完整训练
- 中型（100M-1B）能微调（LoRA / QLoRA）· 但 pretrain 真起来还是要云
- 7B+ 模型纯<em>推理</em>用 GGUF + llama.cpp 在 M2/M3 上很顺

---

## 🛠 硬件梯度建议

| 你有什么 | 能做 phase | 不能做 |
|---|---|---|
| 只有 Mac (M1/M2/M3 · 16GB+) | 0/1/2/4-LoRA/5/6-LoRA/7/9 | phase 3 真 pretrain · phase 8 多卡 |
| Mac + 云端 GPU 偶尔租 | 全部 | — |
| 1×NVIDIA 24GB+ (3090/4090/A100) | 全部 · 但大模型还是要 LoRA | 8×卡的 FSDP/DeepSpeed 验证 |
| 多卡 / 集群 | 全部 + 真 pretrain GPT-2 size | — |

---

## 🔗 与 taotao-agent 的关系

```
taotao-llm  ─────训出 base / SFT 模型─────►  taotao-agent
   (训练侧)                                       (应用侧)
```

phase 9 deploy 完之后 · 把 GGUF 喂给 taotao-agent 的 Ollama backend · 你就能
跟<strong>自己训出来</strong>的 agent 对话 —— 这是这两个项目合在一起的终极目标。

---

## 📖 推荐先读

学之前可以扫一下，会建直觉：

- **Karpathy · "Let's build GPT: from scratch" YouTube** · 我们的 phase 1 就是它的硬核版
- **Sebastian Raschka · "Build a Large Language Model (from scratch)"** 书 · 跟本仓库节奏 90% 重合
- **Anthropic · "Soul of a New Machine"** 讲 LLM 训练直觉
- **Hugging Face course** · 跟 industrial 轨技术栈一致
- 配套 [`taotao-agent/docs/post-training.html`](https://github.com/doraemonlyz-jpg/taotao-agent/blob/main/docs/post-training.html) · SFT / DPO / RLHF 的概念扫盲

---

## 🤝 Contributing

本仓库定位是<strong>学习项目</strong>而非 production 库。但欢迎：

- typo / fact 修正
- 你跑通的训练实验 + log 分享
- 中文术语翻译完善
- 给某 phase 加更好的可视化 notebook

参见 `CONTRIBUTING.md`（待加）。

---

## 📜 License

MIT · see [`LICENSE`](LICENSE)

---

— taotao
