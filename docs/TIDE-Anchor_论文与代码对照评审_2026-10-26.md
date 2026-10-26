**TIDE-Anchor 论文与代码对照评审**

评审日期：2026-10-26。对象为本次粘贴的 LaTeX 方法章节、当前工作区源码、相关测试及仓库实验稿。本次未修改引擎实现和论文原稿。

**总体判断**

这版章节的架构主线成立，且与工程有较好的对应关系：Context Ledger 提供运行中的任务状态，Memory Store 提供相关历史，原文归档提供回溯入口，Hook 将这些组件接入图执行。尤其是“稳定任务供给”和“选择性证据供给”的区分，适合作为本节的核心贡献。

但论文需要明确区分三个层次：设计希望保证什么、当前组件实际执行什么、实验具体验证了什么。目前最需要修正的是预算保证、事实验证含义、目标完整性和恢复编排边界。现有代码足以支撑一个上下文治理系统的设计章节，尚不足以无条件支撑“原始目标完整保留”“每个最终模型请求严格受预算约束”“检查点可自动恢复全部运行信息”等强保证。

**一、论文各部分与代码的对应程度**

| 论文内容 | 判断 | 需要保留的限定 |
|---|---|---|
| 账本与历史记忆分离 | 对应充分，是最清晰的主线 | 启用账本后，锚点生成不依赖检索命中；模型是否使用它仍取决于节点及适配器 |
| Ledger 保存目标、约束、计划、事实、失败 | 数据结构及持久化均存在 | 初始目标被截断；动态约束和 TODO 的推进并非全部自动完成 |
| WORKING/TASK/PROJECT/GLOBAL 四作用域 | 实现存在 | 依赖正确传递作用域 ID；存储生命周期不完全等于逻辑作用域生命周期 |
| 作用域与存储介质独立 | 表述正确 | 有默认路由，也允许 metadata 覆盖；某些可选后端主要承担写入或镜像 |
| 摘要与原文引用双轨 | 实现存在 | 默认摘要是字段选择和截断；存在账本与记忆两套归档根目录 |
| 四种唤醒等级 | 表格与默认值一致 | 配置驱动；Silent 仍会检索，Recovery 的时态项主要是新近度加权 |
| 标准唤醒跨作用域统一排序 | 与 wake 路径一致 | cascade_read 是另一种逐层补足接口，不能混用二者的解释 |
| 超限后 budget_paused | 实现且测试通过 | 核心运行时检查共享状态，发生在本步记忆和锚点注入之前 |
| 漂移检测与 TODO 建议 | 总体准确 | 检测与纠偏分开；默认记录建议，模型判断需实际配置可用 Judge |
| 上下文/图检查点职责分离 | 设计及接口存在 | 缺少统一、原子化配对和自动 Recovery 唤醒；暂停检查点还存在具体缺陷 |
| 唤醒复杂度公式 | 在给定假设下合理 | 当前是作用域内候选读取、逐项评分和完整排序，不是 ANN 检索 |

**二、实际执行链路**

入口可从 [orchestrator.py:299](D:/Pro/Agent-Graph/Long_Task_1/engine/orchestrator.py:299) 的 set_context_policy 开始理解：它建立账本、预算控制器、注入器和漂移检测器；set_memory 另外配置记忆后端；构图时由 _build_hook_manager 将这些组件交给 HookManager。

~~~mermaid
flowchart TD
    S["Graph：开始一个执行超步"] --> C["可选：保存本步之前的图检查点"]
    C --> L["Hook.on_step_start：更新 Ledger"]
    L --> B["检查当前共享状态的粗略预算"]
    B -->|超限| P["设置 budget_paused，退出执行循环"]
    B -->|通过| W["各节点 on_node_start：检索并格式化历史记忆"]
    W --> A["从 Ledger 构造任务锚点"]
    A --> N["节点/适配器构造输入并执行"]
    N --> V["评价节点输出"]
    V --> U["更新 Ledger、摘要、原文引用及事实状态"]
    U --> D["漂移检测与 TODO 更新建议"]
    D --> M["节点输出写入 TASK 记忆"]
    M --> G["按验证决策合并状态，确定下一前沿"]
    G --> S
~~~

上述顺序的直接依据是 [hooks.py:230](D:/Pro/Agent-Graph/Long_Task_1/engine/hooks.py:230)、[hooks.py:595](D:/Pro/Agent-Graph/Long_Task_1/engine/hooks.py:595)、[hooks.py:279](D:/Pro/Agent-Graph/Long_Task_1/engine/hooks.py:279) 和 [graph.py:330](D:/Pro/Agent-Graph/Long_Task_1/engine/graph.py:330)。

论文的 C_s=A(L_s)⊕I_s⊕E_s⊕O_s 可以保留为逻辑组成式。它不应被解释为所有适配器都采用相同的消息顺序，也不代表最终请求只含这四部分。实际 AgentRuntime 还可能加入短期会话、技能、知识库内容及上游消息。

**三、最需要收紧的论文表述**

**1. 预算检查约束的是共享状态估计值，尚未形成最终 Prompt 的硬保证。**

[budget.py:54](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/budget.py:54) 对输入对象序列化后估算用量，核心近似为字符数除以 4；max_context_tokens≤0 时不限制。论文给出的有效输入预算公式与代码一致。

但 [hooks.py:235](D:/Pro/Agent-Graph/Long_Task_1/engine/hooks.py:235) 在步骤开始时检查状态；[hooks.py:244](D:/Pro/Agent-Graph/Long_Task_1/engine/hooks.py:244) 随后才为节点唤醒记忆并生成账本注入块；[agent_runtime.py:346](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/agent_runtime.py:346) 更晚才构造真实模型输入。当前没有统一的最终请求预算闸门。

补充示例将预算设为 100，初始状态估计为 21；注入 6,162 字符的记忆后，节点仍执行成功。节点可见状态的估计值变为 6,651。后者包含结构化记忆等额外字段，不能当成实际模型输入 token 数，但该示例确实证明了注入后没有再执行同一预算检查。

此外，max_memory_tokens 和 max_tool_summary_tokens 在预算对象中被保存，check 并未分别执行这两个子预算。仓库 context_policy.yaml 未指定最大上下文值，继承的默认值为 0。

建议在论文预算段落补一句：

> 当前运行时在步骤开始阶段对共享状态进行粗粒度 Token 估算，并据此实施超限暂停。该估算用于运行治理；节点完成记忆注入及适配器输入装配后，仍需结合目标模型的分词器和完整请求结构进行最终预算校验。

**2. “已验证事实”目前主要表示通过了配置的节点输出评价。**

[ledger.py:154](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/ledger.py:154) 将压缩后的输出摘要作为一个 ContextFact 写入，verified 来自节点评价结果。默认 [evaluation.py:57](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/evaluation.py:57) 主要检查必要字段、必要文件及父级确认，没有通用的外部事实核验。没有配置额外条件的合法字典输出通常可以通过。

因此，字段存在、文件存在、节点结果被接受，与某个事实在真实世界中正确，是不同的判断。

建议增加定义：

> 账本中的验证状态表示该条结果是否通过配置的节点输出评价或外部确认，其可信程度取决于验证器及证据条件；默认规则评价不等价于独立的事实真实性证明。

“提取关键事实”也宜限定为“根据结构化输出或摘要形成候选事实记录”。默认流程没有普遍执行语义级实体、关系、数值和时间抽取；单独配置时态证据模块后，才存在更丰富的事实结构。

**3. “稳定供给”成立，但原始目标的完整性和动态更新有边界。**

[ledger.py:205](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/ledger.py:205) 从 original_goal/goal/task/input/query 选择初始目标，再调用 _clip。默认 summary_max_chars=600，因此 original_goal 字段并不保证保存完整的原始请求。

补充示例输入 675 字符，在第 650 字符之后放置关键要求，结果账本只保存 600 字符，尾部要求丢失。它与“目标可能较长、关键限制位于末尾”的任务场景直接相关。

同时，on_step_start 主要更新执行前沿与 next_action；on_node_end 更新 completed_steps、pending_steps、摘要和候选事实。节点完成不自动等价于 active_todo 完成。TODO 的语义状态通过 TodoManager 或受限自动更新接口推进；新增约束也不能仅靠往任意 state 字段写入就假定账本已同步。

建议把“原始目标”改为“初始化时保存的目标描述”，并补充长度策略；或者在工程中同时保存完整不可变目标及其可注入短版本，再支持严格的完整保留表述。

**4. 压缩和归档有实现，但并不自动缩减所有共享历史。**

[compressor.py:81](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/compressor.py:81) 的摘要策略是优先提取 output/result/answer/input 等字段，再截断字段内容。它是确定性的文本压缩，不能直接写成自动理解并保留全部关键语义。

该压缩器为 Ledger 生成摘要和归档，不会直接重写或裁剪原始 GraphState.messages。[agent_runtime.py:380](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/agent_runtime.py:380) 在没有定向 capsule 时，仍可能把共享消息历史加入模型输入。因此，摘要写入账本与所有运行历史都已受控压缩，应分别论述。

可使用：

> 系统对需复用的节点结果建立摘要及原文回溯入口，减少相关历史证据在后续调用中的重复展开。共享状态和适配器内部会话的裁剪需由相应上下文策略另行控制。

**5. 漂移检测提供信号，自动纠偏需要明确控制方。**

[drift.py:188](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/drift.py:188) 检测重复节点、待执行步骤积压、重复摘要、资源引用和文件操作，并可启用 embedding/LLM 目标偏离检查。默认重复摘要判断使用归一化指纹相等，不能将所有规则项都理解成语义模型检测。

[hooks.py:718](D:/Pro/Agent-Graph/Long_Task_1/engine/hooks.py:718) 将结果写入状态和账本失败摘要，默认不会仅因 drifted=True 就自动暂停或重规划。todo_update_mode=suggest 时，建议写入 open_questions；auto 模式仅应用 insert、replace、cancel、complete、select 等支持动作。

你当前正文已经写明“具体恢复动作由对应控制流程决定”，这点准确，建议保留。

**6. 检查点保存能力存在，统一恢复协议尚需调用方落实。**

上下文检查点复制 Ledger、MEMORY.md 和当前账本工具摘要关联的原文归档。图检查点保存 state、step 和 frontier。两者没有统一的配对清单或事务边界；Graph.resume_from 不会自动调用 ContextCheckpointStore.restore，也不会自动把唤醒等级切换到 Recovery。

特别要说明：上下文检查点不会自动备份整个 Memory Store 的 SQLite 索引、PROJECT 记忆归档及所有可选后端。恢复历史检索能力仍依赖这些外部存储保持可用，或另有备份方案。

恢复旧图检查点时，Hook 还会从磁盘加载对应 run 的账本；若磁盘账本已推进到更晚版本，就需要显式恢复匹配的上下文检查点，避免新旧状态混用。

当前正文的“双检查点分工”可以保留，但图中的完整恢复顺序应标为恢复编排协议，并交代实际调用方，避免被理解为单个 resume 接口自动完成全部动作。

**四、代码文件导读**

| 文件 | 关键对象/入口 | 作用及阅读重点 |
|---|---|---|
| [context/_types.py:266](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/_types.py:266) | ContextLedger | 任务状态的数据模型；ContextFact、ToolSummary、TodoItem、TodoEvent、ContextBudget 在同文件定义 |
| [context/ledger.py:22](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/ledger.py:22) | ContextLedgerStore | 按 run 保存 ledger.json；更新节点进度、摘要、候选事实和失败；同时刷新 MEMORY.md |
| [context/renderer.py:14](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/renderer.py:14) | render_markdown | 将账本渲染为可审计文档；MEMORY.md 是账本的可读视图，不是另一个独立语义记忆库 |
| [context/injector.py:90](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/injector.py:90) | ContextInjector.build | 构造面向节点的任务锚点；默认各选最近 8 条 verified/unverified 记录，加入活动/相邻/已完成 TODO |
| [context/todo.py:11](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/todo.py:11) | TodoManager | 管理待办状态、活动项和修订事件；一次节点成功与一个语义 TODO 完成需要显式关联 |
| [context/compressor.py:26](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/compressor.py:26) | compress_update | 长输出保存为 raw/run/step_node_hash.json；摘要和 raw_ref 写回账本；默认长度阈值为字符数 |
| [context/budget.py:33](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/budget.py:33) | ContextBudgetController | 粗略预算估算和允许/暂停决策；它本身不实施自动压缩 |
| [context/drift.py:142](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/drift.py:142) | DriftDetector | 规则停滞信号及可选目标偏离检测；检测结果还需控制流程消费 |
| [context/policy.py:17](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/policy.py:17) | ContextPolicy | 读取平坦 YAML/JSON，构造上述组件；应区分策略字段、默认值和实际启用条件 |
| [memory/_types.py:110](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/memory/_types.py:110) | wakeup_profile | 定义四种唤醒配置，以及作用域、介质、记忆项等结构 |
| [memory/store.py:53](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/memory/store.py:53) | HybridTieredMemoryStore | 记忆写入路由、SQLite 持久化、跨作用域检索、摘要与归档展开，是记忆主实现 |
| [memory/embedding.py:19](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/memory/embedding.py:19) | HashingEmbeddingModel / BGEM3EmbeddingModel | 默认是依赖较少的哈希向量；BGE-M3 需要显式配置，不能把默认 dense 等同于 BGE-M3 |
| [memory/_utils.py:32](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/memory/_utils.py:32) | _sparse_score / _recency_score | 词项覆盖和新近度计算；这里的 sparse 并非 BM25 或 BGE 稀疏向量 |
| [memory/temporal.py:95](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/memory/temporal.py:95) | TemporalEvidenceMemoryStore | 独立的有效期、事实历史、冲突和证据管理；不能由普通 Recovery 配置自动推导出全部这些能力 |
| [hooks.py:159](D:/Pro/Agent-Graph/Long_Task_1/engine/hooks.py:159) | HookManager | 将账本、记忆、评价、漂移和恢复接入图生命周期；理解整体执行顺序的首要文件 |
| [graph.py:296](D:/Pro/Agent-Graph/Long_Task_1/engine/graph.py:296) | CompiledGraph.astream | 实际推进 step/frontier、调用 Hook、处理暂停、合并节点状态和恢复图检查点 |
| [adapters.py:89](D:/Pro/Agent-Graph/Long_Task_1/engine/adapters.py:89) | make_agent_node_func | 对字符串输入拼接锚点、技能和记忆；已有 Msg 等非字符串输入走不同分支 |
| [agent_runtime.py:346](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/agent_runtime.py:346) | _build_prompt | 平台 Agent 的真实输入装配；检查账本最终是否送入模型时必须读到这里 |
| [context/checkpoint.py:51](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/checkpoint.py:51) | ContextCheckpointStore | 创建和恢复任务信息及账本关联归档快照 |
| [engine/checkpoint.py:51](D:/Pro/Agent-Graph/Long_Task_1/engine/checkpoint.py:51) | GraphCheckpointStore | 保存执行状态与前沿；与 context/checkpoint.py 职责不同 |
| [experiments/long_task.py:248](D:/Pro/Agent-Graph/Long_Task_1/engine/experiments/long_task.py:248) | _run_one | 实验专用多轮治理循环；不是生产 Graph+Hook 路径的完整复用 |

推荐阅读顺序：ContextLedger 数据结构 → LedgerStore → Injector → HookManager → Graph → AgentRuntime → MemoryStore → 两种 Checkpoint → 实验驱动。

两类归档尤其容易混淆：账本原文通常位于账本根目录的 raw/ 下，存 JSON 节点更新；Memory Store 的原文位于它自己的 project_archive/ 下，存带元数据的 Markdown。Hook 目前分别写入账本和 TASK 记忆，并未把它们实现为一个共享原文仓库。raw_ref 必须结合所属存储的根目录解析。

**五、唤醒策略和检索算法的精确解释**

| 等级 | 范围 | K | 检索 | 展开 |
|---|---|---:|---|---|
| Silent | WORKING、TASK | 3 | dense | 不展开 |
| Standard | 四作用域 | 5 | hybrid | 注入摘要与 ref，保留显式展开接口 |
| Deep | 四作用域 | 8 | hybrid | 前 2 个命中项在有 raw_ref 时展开 |
| Recovery | 四作用域 | 10 | hybrid_temporal | 所有有 raw_ref 的命中项尝试展开 |

每条展开文本在格式化时截取至 4,000 字符；文件先完整读取，再截取，故长度上限不能约束磁盘读取量。Standard 不会仅因提示中出现“按需展开”就自动提供模型工具调用，需要调用方使用 expand 等接口。

当前普通混合评分为：

S_hybrid(q,m) = 0.65·Dense(q,m) + 0.35·Sparse(q,m)。

Recovery 额外增加 0.15·Recency(m.ts)。importance 和 ts 还作为排序的后续比较键。新近度是写入时间衰减，不等于按事实有效期回答“某时刻的事实”。

_read_scopes 收集所选作用域的候选，统一排序后按记忆 ID 去重；不是按内容或 raw_ref 语义去重。cascade_read 则从窄作用域开始，数量满足就停止。你贴出的版本已经正确选择了前一种解释。

当前 SQLite 的语义索引表不等于已使用向量数据库 ANN 搜索：查询主要按 scope/scope_id 取回候选，再在 Python 里评分和排序。缺少对应 scope_id 时，读取范围会扩大，因此正确传递 run/project/global 标识是隔离前提。

论文 T_ret(N)+O(N log N+KR) 的表达合理。若进一步展开，可说明 T_ret 包含候选读取、向量维度相关的相似度计算和词项处理；模型辅助漂移判断还存在独立调用延迟，完整归档读取和检查点复制也应单列。

**六、实现中已复现的具体缺陷**

| 优先级 | 位置与触发条件 | 观察结果 | 对论文的影响 |
|---|---|---|---|
| 高 | [hooks.py:235](D:/Pro/Agent-Graph/Long_Task_1/engine/hooks.py:235)：小状态先通过预算，节点随后注入大量记忆 | 超过原先限制的节点上下文仍可执行 | 不能把步骤级状态检查当成最终请求硬预算保证 |
| 高 | [graph.py:533](D:/Pro/Agent-Graph/Long_Task_1/engine/graph.py:533)：因 budget_paused 退出循环且启用了图检查点 | final 检查点仍写 status=completed、frontier=[]，并成为 latest | 默认恢复 latest 会丢失待执行位置；需修正暂停保存语义 |
| 高 | [hooks.py:378](D:/Pro/Agent-Graph/Long_Task_1/engine/hooks.py:378)：带 reasoning subtask 的节点走异常处理 | 引用了函数内未定义的 evaluation，触发 NameError；后续还引用未定义 update | 该组合配置下异常记录和恢复路径会被二次异常打断 |
| 中 | [ledger.py:216](D:/Pro/Agent-Graph/Long_Task_1/engine/modules/context/ledger.py:216)：目标超过默认字符限制 | 原始目标尾部被截断 | “完整原始目标稳定保存”缺少实现支撑 |

这些缺陷是在现有测试通过后通过补充最小示例确认的。它们不否定架构分工，但会影响对可靠性和保证范围的表述。本次按分析任务范围未修改这些实现。

**七、实验能支撑哪些结论**

你本次粘贴的正文主要是方法，没有直接启用早期注释中的性能数字，这种写法更稳妥。仓库另一个实验稿写有 Goal Retention=0.90、Recovery=0.91、平均上下文=954 等结果，但本工作区没有它引用的 long_task_multiturn_bge_100_v12 结果目录，无法在本次核验逐任务日志和汇总值。现有旧版实验目录采用其他协议，不应据此验证或否定 v12 数字。

可以确认 v12 数据文件存在，含 100 个唯一任务 ID，来自 10 个 trajectory；gold evidence 数量为 4 条任务各 1 条、96 条任务各 2 条。SHA-256 与仓库实验稿一致：

c102dd765d0eb054a3bb0bab90c99b3b12b50b9278644ab09edfb84d5d7b6414

更关键的是区分以下实验专用处理和核心运行时：

- [long_task.py:301](D:/Pro/Agent-Graph/Long_Task_1/engine/experiments/long_task.py:301) 用已知事件中的 proposed_goal 与原始目标的相似度产生漂移防护提示，未完整复用通用 DriftDetector 的实际检测路径。
- [long_task.py:620](D:/Pro/Agent-Graph/Long_Task_1/engine/experiments/long_task.py:620) 对已装配的上下文检查预算，并实施轨迹压缩和优先级截断；这些逻辑比核心 Hook 当前的状态级暂停更完整。
- [long_task.py:258](D:/Pro/Agent-Graph/Long_Task_1/engine/experiments/long_task.py:258) 在任务开始时检索历史，随后多轮复用；它不能单独证明每节点自适应查询和四级唤醒策略的效果。
- [long_task.py:399](D:/Pro/Agent-Graph/Long_Task_1/engine/experiments/long_task.py:399) 将实验 Runtime 写入上下文检查点 metadata；恢复实验状态不等价于验证生产图的 frontier 恢复。
- LongTaskExperimentConfig 数据类默认 qa_solver=extractive，而 CLI 默认 llm；只有核验实际运行配置、调用及回退记录，才能把一轮实验称为真实 LLM 多轮评测。模型步骤和轨迹裁判异常均有启发式回退，正式报告应给出回退率。
- 平均上下文采用粗略估算，并排除共享系统提示词；Memory Use Accuracy 在代码中是“证据充分且回答正确”的联合比例，不是以充分证据任务数为分母的条件正确率。

LoCoMo 原始工作评测长对话记忆，包含 QA、事件摘要和多模态对话等任务。[LoCoMo 原论文](https://aclanthology.org/2024.acl-long.747/)

补充更正（2026-10-26 实验核验）：当前 v12 的 100 条全部为 category=1。项目将其称为“直接事实题”，但官方评测代码把 category=1 定义为 multi-hop；因此应称“按证据数量与检索锚点筛选的多跳子集”。先前版本沿用项目命名的“直接事实”表述不准确。加上固定六轮治理事件后，它属于 LoCoMo 派生多轮治理评测，不宜外推到完整 LoCoMo 或任意超长执行链。[官方类别判分代码](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py)

建议实验章节至少补齐：模型与运行参数、数据哈希、逐任务原始结果；Ledger/压缩/漂移/检查点消融；真实 Graph+Hook 集成恢复案例；长执行步数和多次中断设置；完整请求 token 统计。由于 100 个问题共享 10 组源对话，置信区间宜考虑按源对话聚类，避免把所有问题视为完全独立样本。

**八、学术定位及可直接采用的贡献表述**

有限窗口中的分层记忆和外部存储管理已有先行研究，MemGPT 就讨论了快慢记忆层之间的数据移动及控制流。因此，不宜将“分层存储+摘要+检索”整体直接表述成全新原理。[MemGPT 原论文](https://arxiv.org/abs/2310.08560)

本项目更有辨识度的部分是：把当前任务状态与历史检索证据分开组织，并通过图生命周期，把信息更新、评价、漂移记录、预算状态与恢复接口联系起来。能否形成强研究贡献，取决于对这套组合机制的清晰定义，以及相对单纯记忆检索的可复现实验增益。

引言中“长历史不等于有效利用”的动机有研究依据；Lost in the Middle 表明相关信息在上下文中的位置会影响模型利用效果。该引用支持问题动机，不直接证明本项目机制已解决全部长上下文失效问题。[Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)

可将本节贡献收束为：

> 本文面向图驱动的长程智能体任务，设计任务状态与历史证据分离的上下文管理机制。系统通过运行账本持续提供目标、约束和进度，通过分层记忆与原文引用选择性提供历史信息，并将状态更新、预算判定、漂移记录和检查点接口接入节点生命周期。该机制为上下文的审计、调整和恢复编排提供统一的状态依据。

若要进一步形式化，可补定义账本更新函数 L_(s+1)=U(L_s,ΔX_s,V_s)，其中 V_s 为评价结果；再写清目标与硬约束哪些字段保持不变、哪些更新必须有修订事件。这样比增加概念性形容词更能增强方法章节的可检查性。

**验证记录**

执行现有测试：

~~~text
pytest tests/test_context_ledger.py tests/test_memory_store.py tests/test_graph_checkpoint.py -q -k "not test_bge_m3_embedding_model_produces_vectors" -p no:cacheprovider
61 passed, 1 deselected in 4.48s
~~~

排除项会实际加载 BGE-M3 模型权重。本次未重新运行真实模型/API 实验。补充示例使用临时目录，验证结果保存在同目录的 JSON 文件中。
