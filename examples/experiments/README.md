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
