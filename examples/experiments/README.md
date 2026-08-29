# 实验入口说明

仅保留四组主实验：多 Agent 编排稳定性、长期记忆端到端 QA、长期记忆与上下文治理联合、技能闭环。完整的数据准备、运行和结果解释见 `docs/8.29实验3跑起来步骤.md`。

所有公开数据集先通过 `prepare_datasets.py` 规范化到 `data/processed/`，然后运行对应 runner。长期记忆的正式结果必须使用默认的 `--qa-solver llm`；`extractive` 仅用于离线冒烟测试。
