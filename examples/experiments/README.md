# 实验入口说明

本目录提供科研实验的命令行入口，核心逻辑在 `engine.experiments` 包中。

## 记忆实验

LongMemEval 主实验：

```bash
python examples/experiments/run_memory_experiment.py --dataset data/longmemeval.jsonl --source longmemeval
```

LoCoMo 补充实验：

```bash
python examples/experiments/run_memory_experiment.py --dataset data/locomo.json --source locomo --limit 50
```

mem0 真实对照实验：

```bash
python examples/experiments/run_memory_experiment.py --dataset data/longmemeval.jsonl --source longmemeval --backend mem0
```

mem0 会复用 `.env` 中的 DeepSeek 云端模型作为 LLM，并使用本地 BGE-M3 + Qdrant 做
embedding 与向量检索。

正式对照组可通过 `--backend no_memory/full_context/engine/mem0` 选择；本项目后端还支持
`--no-llm-judge`、`--retrieval-mode`、`--project-only` 和 `--append-only`。

最小记忆消融矩阵：

```bash
python examples/experiments/run_memory_ablation.py --dataset data/longmemeval.jsonl --source longmemeval
```

## 长任务状态治理联合实验

一次运行 Plain、Full-Context、Memory Only、Context Only、Ours Full 五组：

```bash
python examples/experiments/run_long_task_experiment.py --size 50 --method all
```

该 runner 是确定性的后端机制实验，验证记忆、Ledger、注入、漂移、预算和 checkpoint
能否联合保留所需信号；它不等同于真实 LLM solver 的最终问答评测。

## 技能实验

SkillEvolBench 小子集：

```bash
python examples/experiments/run_skill_experiment.py --dataset data/skillevolbench_sample.jsonl --limit 30
```

## 编排实验

生成并运行 300 条小型 workflow：

```bash
python examples/experiments/run_workflow_experiment.py --size 300
```

## 端-边-云路由实验

```bash
python examples/experiments/run_routing_experiment.py
```

默认输出 All Device、All Cloud、Heuristic、Learned 四组。准确率和敏感任务误送云率
来自真实决策；成本与端/云推理延迟默认是可配置代理值，正式论文应替换为设备 trace 实测值。
