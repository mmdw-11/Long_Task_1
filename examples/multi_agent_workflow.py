"""示例：subagent 层级 + graph 连接 混合工作流（真实 OpenAI 调用，配置来自 .env）。

准备：
1. 复制 .env.example 为 .env，并填入 OPENAI_API_KEY（OPENAI_BASE_URL / OPENAI_MODEL 按需）
2. 安装依赖：pip install agentscope openai python-dotenv

运行：
    python examples/multi_agent_workflow.py

工作流说明（混合两种编排维度）：
- **subagent 层级**：先独立创建 researcher / writer，再用 add_sub_agent 把这两个
  “现有 agent”挂到主 agent planner 名下（父子归属关系，供前端可视化分组）。
- **graph 连接**：用图的边定义真正的执行流——planner 扇出到两个子 agent，
  researcher / writer 再扇入到 reviewer（fan-out + fan-in）。

数据流：每个 agent 读取“任务主题 + 此前所有 agent 的发言”作为上下文（类似群聊），
输出累积到共享状态的 messages 中。
"""

import asyncio

from engine import Orchestrator
from engine.config import build_openai_agent, load_settings
from engine.node import Node, NodeType


def make_openai_node_factory(settings):
    """构造一个 node_factory：依据 AgentSpec 生成“真实 OpenAI agent”图节点。

    节点执行逻辑：把任务主题与历史发言拼成上下文喂给 agent，输出追加到 messages。
    """
    from agentscope.message import Msg, TextBlock

    def factory(spec) -> Node:
        agent = build_openai_agent(spec.name, spec.sys_prompt, settings)

        async def node(state):
            topic = state.get("input", "")
            history = state.get("messages", []) or []

            lines = []
            if topic:
                lines.append(f"任务主题：{topic}")
            for m in history:
                lines.append(f"[{m['agent']}] {m['text']}")
            prompt = "\n".join(lines) if lines else topic

            reply = await agent.reply(
                Msg(name="user", content=[TextBlock(type="text", text=prompt)], role="user")
            )
            text = reply.get_text_content()
            return {
                spec.name: text,
                "messages": [{"agent": spec.name, "text": text}],
            }

        return Node(spec.name, node, NodeType.AGENT, metadata={"id": spec.id, "model": spec.model})

    return factory


async def main() -> None:
    settings = load_settings()
    print(f"使用模型：{settings.model}"
          + (f"（base_url={settings.base_url}）" if settings.base_url else ""))

    orch = Orchestrator()

    # 1) 主 agent
    planner = orch.create_agent(
        "planner", sys_prompt="你是项目协调者，负责把用户主题拆解为写作要点与调研方向，简明列出。"
    )

    # 2) 先独立创建两个“现有 agent”，再把它们挂为 planner 的子 agent（层级关系）。
    #    auto_connect=False：只建立父子归属，执行边稍后由图显式定义。
    researcher = orch.create_agent(
        "researcher", sys_prompt="你是资料研究员，围绕要点给出关键事实与论据，简洁分条。"
    )
    writer = orch.create_agent(
        "writer", sys_prompt="你是作家，综合上面的要点与资料，写一段连贯流畅的文字。"
    )
    orch.add_sub_agent(planner, researcher, auto_connect=False)
    orch.add_sub_agent(planner, writer, auto_connect=False)

    # 3) 一个独立的评审 agent（非子 agent）。
    reviewer = orch.create_agent(
        "reviewer", sys_prompt="你是主编，综合前面所有内容给出最终定稿，并附一句总体点评。"
    )

    # 4) 用图的边定义执行流：planner 扇出到两个子 agent，二者再扇入 reviewer。
    orch.set_entry(planner)
    orch.connect(planner, researcher)
    orch.connect(planner, writer)      # fan-out
    orch.connect(researcher, reviewer)
    orch.connect(writer, reviewer)     # fan-in（reviewer 只执行一次）

    # 5) 展示混合结构：子 agent 层级 + 图连接。
    print("\n=== subagent 层级 ===")
    print(f"planner 的子 agent: {[orch.get_agent(c).name for c in orch.get_agent(planner).children]}")
    print("\n=== 图结构（节点 + 边）===")
    print(orch.build_graph(node_factory=make_openai_node_factory(settings)).to_dict())

    # 6) 真实执行。
    compiled = orch.build_graph(node_factory=make_openai_node_factory(settings))
    state = await compiled.ainvoke({"input": "人工智能与教育"})

    print("\n=== 各 agent 产出 ===")
    for m in state.get("messages", []):
        print(f"\n【{m['agent']}】\n{m['text']}")


if __name__ == "__main__":
    asyncio.run(main())
