# engine

在 **[AgentScope](https://github.com/agentscope-ai/agentscope)** 之上扩展的、
**类 LangGraph** 的 Agent 图编排框架，并额外提供一套**面向前端可视化**的易用接口
与 REST 服务。

AgentScope 原生只提供 `sequential_pipeline` / `fanout_pipeline` / `MsgHub` 等线性编排；
本项目补齐了 LangGraph 式的**图编排**能力（节点、边、条件边、状态、循环），
并在此基础上构建了**可插拔扩展体系**（记忆、动态路由、流控、恢复策略、资源调度）
和**钩子机制**，同时把这些能力封装成可供前端拖拽可视化的 API。

---

## 功能特性

### 1. LangGraph 式图编排（`StateGraph` / `CompiledGraph`）
- **节点（Node）**：可包裹普通函数（同步/异步）、AgentScope agent、或子图。
- **静态边** `add_edge(a, b)`：顺序连接，支持一对多**并行扇出**。
- **条件边** `add_conditional_edges(src, fn, path_map)`：根据状态动态路由（分支/循环）。
- **`START` / `END`** 哨兵节点，`set_entry_point` / `set_finish_point` 语法糖。
- **共享状态 + 归并器（reducer）**：默认覆盖，`append_reducer`/`add_messages` 可累积消息（多 agent 汇聚）。
- **Pregel 风格超步执行**：天然支持顺序、分支、并行、循环，并带**递归步数上限**防止死循环。
- **流式执行** `astream()`：逐节点产出 `node_start`/`node_end`/`final` 事件，便于前端实时展示。

### 2. 可插拔扩展体系（`engine.modules` + `engine.hooks`）

五大扩展模块均提供 **ABC 接口 + NoOp 空实现桩**，未接入时零开销，按需插拔替换：

| 模块 | 文件 | 能力 |
|------|------|------|
| **记忆** | `modules/memory.py` | 四级层级记忆（WORKING → TASK → PROJECT → GLOBAL），级联检索 |
| **动态路由** | `modules/routing.py` | 静态 Router + 运行时 RoutingPolicy，运行时动态决定后继节点 |
| **流控** | `modules/flow.py` | 依赖解析 + 条件激活，决定节点执行/跳过/延迟 |
| **恢复策略** | `modules/recovery.py` | 依据失败轨迹决定修复路径（重试/跳转/中止） |
| **资源调度** | `modules/scheduling.py` | 端边云资源申请与释放 |

**钩子机制**（`hooks.py`）：
- `ExecutionHook`：定义所有扩展点（`on_step_start`、`on_node_start`、`on_node_end`、`on_node_error`、`resolve_successors` 等），全部默认 no-op。
- `HookManager`：把各扩展模块桥接到引擎回调，未注册模块时行为与原始引擎一致。

**失败轨迹**（`failure.py`）：
- `FailureRecord` / `FailureTrace`：记录图执行过程中各节点的失败信息，作为动态路由、恢复策略等模块的共享上下文。
- 约定以 `__failures__` 字段（append reducer）承载于 `GraphState` 中。

### 3. 面向前端的编排接口（`Orchestrator`）
- `create_agent(...)` —— 创建 agent，返回 id
- `add_sub_agent(parent, ...)` —— 给某个 agent 添加子 agent（可多次，添加多个）
- `connect(a, b)` / `connect_conditional(...)` —— 连接两个 agent（含条件边）
- `set_entry(id)` —— 指定入口
- `to_dict()` / `from_dict()` —— 结构序列化，供前端保存/加载/渲染
- `build_graph()` —— 一键编译为可执行图
- `set_memory()` / `set_router()` / `set_routing_policy()` / `set_flow_controller()` / `set_recovery_strategy()` / `set_scheduler()` —— 注入可插拔模块

### 4. REST 服务（FastAPI）
把上述能力暴露为 HTTP 接口，前端可直接调用完成可视化编排。

---

## 安装

> AgentScope 2.x 需要 **Python >= 3.10**。推荐用 conda 环境。

```bash
# 安装全部依赖：
pip install -r requirements.txt

# 或按分组安装：
pip install -e ".[agentscope]"   # 接入真实 AgentScope agent（OpenAI）
pip install -e ".[server]"       # 启动 REST 服务
pip install -e ".[test]"         # 运行测试
pip install -e ".[examples]"     # travel_planner 示例工具（DuckDuckGo + Playwright）
```

安装 Playwright 后还需执行：
```bash
python -m playwright install chromium
```

### 配置模型（.env）

```bash
cp .env.example .env    # 然后编辑 .env 填入你的 key
```

`.env` 字段：

| 变量 | 说明 | 默认 |
|------|------|------|
| `OPENAI_API_KEY` | API Key（必填） | - |
| `OPENAI_MODEL` | 模型名 | `gpt-4o-mini` |
| `OPENAI_BASE_URL` | 兼容端点（自建/中转/vLLM，可选） | 官方 |
| `OPENAI_ORG` | 组织 ID（可选） | - |

代码里通过 `engine.config` 使用：

```python
from engine.config import build_openai_agent
agent = build_openai_agent("writer", "你是一名作家")  # 自动读取 .env
```

---

## 快速开始

### A. 纯图编排

```python
import asyncio
from engine import StateGraph, END, add_messages

graph = StateGraph(schema={"messages": add_messages})
graph.add_node("planner", lambda s: {"messages": ["plan"], "count": 0})
graph.add_node("worker",  lambda s: {"messages": ["work"], "count": s.get("count",0)+1})
graph.set_entry_point("planner")
graph.add_edge("planner", "worker")
# 条件边：循环 3 次后结束
graph.add_conditional_edges(
    "worker",
    lambda s: "loop" if s["count"] < 3 else "done",
    path_map={"loop": "worker", "done": END},
)

state = await graph.compile().ainvoke({"input": "start"})
```

### B. 前端友好的 Orchestrator

```python
from engine import Orchestrator

orch = Orchestrator()
boss = orch.create_agent("coordinator", model="gpt-4o-mini")
r = orch.add_sub_agent(boss, name="researcher")   # 添加子 agent
w = orch.add_sub_agent(boss, name="writer")        # 再添加一个子 agent
rev = orch.create_agent("reviewer")
orch.connect(r, rev)                               # 连接两个 agent
orch.connect(w, rev)
orch.set_entry(boss)

graph = orch.build_graph()                         # 编译执行
state = await graph.ainvoke({"input": "写一篇报告"})
structure = orch.to_dict()                          # 导出给前端渲染
```

### C. 接入可插拔模块

```python
from engine import Orchestrator
from engine.modules.memory import MyMemoryStore       # 自行实现的 MemoryStore
from engine.modules.routing import MyRoutingPolicy    # 自行实现的 RoutingPolicy

orch = Orchestrator()
orch.set_memory(MyMemoryStore())
orch.set_routing_policy(MyRoutingPolicy())
# ... 创建 agent、连接、设置入口 ...
graph = orch.build_graph()                            # 模块自动接入钩子
```

### D. 接入真实 AgentScope agent

```python
from engine import StateGraph, add_messages
from engine.adapters import agent_node

graph = StateGraph(schema={"messages": add_messages})
graph.add_node_object(agent_node("writer", writer_agent))
graph.add_node_object(agent_node("critic", critic_agent))
graph.set_entry_point("writer")
graph.add_edge("writer", "critic")
graph.set_finish_point("critic")
```

---

## REST 服务

```bash
python -m engine.server
# 或： uvicorn engine.server.app:app --reload
# 文档： http://127.0.0.1:8000/docs
```

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/agents` | 创建 agent |
| GET | `/api/agents` | 列出 agent |
| GET | `/api/agents/{id}` | 查看单个 agent |
| DELETE | `/api/agents/{id}` | 删除 agent |
| POST | `/api/agents/{id}/sub-agents` | 添加子 agent（可多次） |
| POST | `/api/connections` | 连接两个 agent（支持条件边） |
| DELETE | `/api/connections` | 断开连线 |
| POST | `/api/graph/entry` | 设置入口 |
| GET | `/api/graph` | 导出图结构（节点+边）供可视化 |
| POST | `/api/run` | 执行编排并返回最终状态 |
| GET | `/api/export` | 导出编排 JSON |
| POST | `/api/import` | 从 JSON 导入编排 |

---

## 示例

### 多 Agent 混合工作流（`examples/multi_agent_workflow.py`）

演示 **subagent 层级 + graph 连接**两种编排维度的混合使用：
planner 扇出到 researcher / writer，再扇入到 reviewer（fan-out + fan-in），
每个 agent 调用真实 OpenAI 模型。

```bash
PYTHONPATH=. python examples/multi_agent_workflow.py
```

### 旅行计划流水线（`examples/travel_planner/`）

7 个 Agent 组成的线性流水线，集成 DuckDuckGo 搜索与 Playwright 网页浏览工具：

```
leader → destination → transport → accommodation → itinerary → budget_review → final_report
```

```bash
PYTHONPATH=. python examples/travel_planner/run.py
PYTHONPATH=. python examples/travel_planner/run.py --request "3人从北京去杭州4天，预算6000"
```

---

## 目录结构

```
engine/
├── constants.py        # START / END 哨兵
├── state.py            # 共享状态 + reducer
├── node.py             # 节点抽象（函数 / AgentScope agent / 子图）
├── graph.py            # 核心图引擎（StateGraph / CompiledGraph）
├── adapters.py         # AgentScope agent → 图节点适配器
├── config.py           # OpenAI 模型配置（从 .env 读取）
├── failure.py          # 失败轨迹数据结构（共享基础）
├── hooks.py            # 执行钩子与钩子管理器（扩展点）
├── orchestrator.py     # 面向前端可视化的高层编排 API
├── modules/            # 可插拔扩展模块
│   ├── memory.py       #   四级层级记忆（WORKING/TASK/PROJECT/GLOBAL）
│   ├── routing.py      #   动态路由（静态 Router + 运行时 RoutingPolicy）
│   ├── flow.py         #   流控（依赖解析 + 条件激活）
│   ├── recovery.py     #   恢复策略（重试/跳转/中止）
│   └── scheduling.py   #   端边云资源调度
└── server/             # FastAPI REST 服务
examples/
├── multi_agent_workflow.py    # 多 Agent 混合工作流示例
└── travel_planner/            # 旅行计划 7-Agent 流水线示例
    ├── prompts.py
    ├── tools.py               # DuckDuckGo 搜索 + Playwright 浏览 + 预算计算工具
    └── run.py
tests/                         # pytest 单元测试
```

---

## 测试

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

---

## 与 LangGraph 的概念对照

| LangGraph | 本项目 |
|-----------|--------|
| `StateGraph(State)` | `StateGraph(schema)` |
| `add_node` | `add_node` / `add_node_object` |
| `add_edge` | `add_edge` |
| `add_conditional_edges` | `add_conditional_edges` |
| `START` / `END` | `START` / `END` |
| `compile()` / `invoke` / `stream` | `compile()` / `ainvoke` / `astream` |
| Annotated reducer | `schema` + `reducer`（`add_messages`） |
