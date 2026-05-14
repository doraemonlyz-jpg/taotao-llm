# frontend/ · 训练监控 + chat with my model

> React 19 + Vite + TypeScript · phase 9 才真起来 · 现在是占位。

## 计划页面

| 路由 | 组件 | 当 phase |
|---|---|---|
| `/dashboard` | `DashboardPage.tsx` · loss / lr / throughput / GPU mem 实时曲线 | phase 3+ |
| `/chat` | `ChatPage.tsx` · 跟自训模型对话 (走 server/) | phase 4+ |
| `/eval` | `EvalPage.tsx` · benchmark 结果可视化 + 跟 baseline 对比 | phase 7+ |
| `/runs` | `RunsPage.tsx` · 历史 run 列表 (从 wandb / 本地 json) | phase 3+ |

## 跟 server/ 的接合

后端用 FastAPI (`server/app.py`) 暴露：

- `GET /runs` · 列已完成 + 进行中
- `GET /runs/:id/metrics` · loss curve · LR · throughput
- `GET /runs/:id/checkpoints` · 列可加载的 step
- `POST /chat` · 拿一个 checkpoint id + prompt · SSE 流式回 token
- `GET /eval/:run_id` · benchmark 分数 + 对比

本仓库的目标之一是<strong>看着自己的 loss 曲线下降</strong>，比 wandb 多了一份成就感。

## 风格借鉴

跟姊妹项目 `taotao-agent` frontend 一致 · "Little Prince" 暗色 · Lora + Outfit 字体。

## 当前状态

phase 0 · bootstrap · 只有这个 README。
