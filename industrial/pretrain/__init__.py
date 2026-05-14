"""industrial.pretrain — 工业级 pretrain pipeline (HF datasets + HF tokenizers)。

模块布局
========
- ``data.py``  : HF 数据集流式拉取 + tokenize + pack + .bin 缓存
- ``model.py`` : 复用 ``nano.gpt.GPT`` · 提供 10M / 30M / 100M 三档 preset
- ``train.py`` : 完整训练循环 · 支持梯度累积 / checkpoint resume / jsonl 日志
- ``run.py``   : CLI · ``python -m industrial.pretrain.run prepare`` 或 ``train``
"""
