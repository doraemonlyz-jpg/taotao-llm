# ────────────────────────────────────────────────────────────────────────
#  taotao-llm · 一站式开发命令
# ────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

# 让 make 在 sub-shell 里继承 .env (非必须 · 但很方便)
ifneq (,$(wildcard .env))
include .env
export
endif

# ============== environment ==============================================
.PHONY: install
install:  ## uv sync 安装 backend deps + dev tools (frontend npm i 在 phase 9 才需要)
	uv sync --extra dev

.PHONY: smoke
smoke:  ## 30 秒 import + device check · 跑通就说明环境 OK
	uv run python -c "import torch; print('torch', torch.__version__); \
print('cuda', torch.cuda.is_available()); \
print('mps', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()); \
import transformers, datasets, peft, trl; \
print('hf stack OK', transformers.__version__)"

.PHONY: download-tiny
download-tiny:  ## 拉一个 ~1MB 的 tiny shakespeare · phase 1 用
	@mkdir -p data/datasets/tinyshakespeare
	@curl -sSL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
		-o data/datasets/tinyshakespeare/input.txt
	@echo "✓ data/datasets/tinyshakespeare/input.txt ($$(wc -c < data/datasets/tinyshakespeare/input.txt) bytes)"

# ============== nano (教学版) ============================================
.PHONY: nano-train
nano-train:  ## phase 1 · 在 tiny shakespeare 上训 toy GPT
	uv run python -m nano.train

.PHONY: nano-tokenize
nano-tokenize:  ## phase 2 · 自训一个 BPE tokenizer
	uv run python -m nano.tokenize

.PHONY: nano-sft
nano-sft:  ## phase 4 · minimal SFT 演示
	uv run python -m nano.sft

.PHONY: nano-dpo
nano-dpo:  ## phase 6 · minimal DPO 演示
	uv run python -m nano.dpo

# ============== industrial (工业版) ======================================
.PHONY: industrial-tokenize
industrial-tokenize:  ## phase 2 · HuggingFace tokenizers · GPT-2 同款配方
	uv run python -m industrial.tokenizer.train_hf

.PHONY: prepare-pretrain-data
prepare-pretrain-data:  ## phase 3 · 拉 TinyStories · tokenize · 缓存到 data/pretrain/
	uv run python -m industrial.pretrain.run prepare \
		--dataset-name roneneldan/TinyStories --max-train-tokens 10000000

.PHONY: industrial-pretrain
industrial-pretrain:  ## phase 3 · 在已 prepare 的数据上训 tiny GPT (M2 air ~30 分钟)
	uv run python -m industrial.pretrain.run train --size tiny --max-iters 3000

.PHONY: pretrain-sample
pretrain-sample:  ## phase 3 · 拿训完的 ckpt 续写 "Once upon a time"
	uv run python -m industrial.pretrain.run sample

.PHONY: scaling-law
scaling-law:  ## phase 3 · 6-run 的 scaling law mini 实验 (~30 分钟)
	uv run python scripts/scaling_law.py --quick

.PHONY: industrial-sft
industrial-sft:  ## phase 4 · trl SFTTrainer
	uv run python -m industrial.sft.run

.PHONY: industrial-lora
industrial-lora:  ## phase 5 · LoRA fine-tune
	uv run python -m industrial.lora.run

.PHONY: industrial-dpo
industrial-dpo:  ## phase 6 · trl DPOTrainer
	uv run python -m industrial.dpo.run

.PHONY: eval
eval:  ## phase 7 · 在选定 model 上跑 lm-eval-harness
	uv run python -m industrial.eval.run

# ============== quality ==================================================
.PHONY: lint
lint:  ## ruff check + format check + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy nano industrial || true   # mypy 在新代码 bootstrap 阶段先 warn-only

.PHONY: format
format:  ## ruff format + ruff check --fix
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: test
test:  ## pytest
	uv run pytest

# ============== frontend / server (phase 9 才用) ========================
.PHONY: front-dev
front-dev:  ## 前端 dev server (Vite, :5181)
	cd frontend && npm install && npm run dev

.PHONY: server-dev
server-dev:  ## FastAPI dashboard backend (:8001)
	uv run uvicorn server.app:app --reload --port 8001

# ============== help =====================================================
.PHONY: help
help:  ## 列出全部命令
	@echo "taotao-llm · 可用命令："
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
