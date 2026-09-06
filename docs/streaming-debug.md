# 智能体与工作流实时调试

## 使用方式

本次修改包含后端代码。停止原后端进程后，在后端目录重新启动：

```powershell
cd D:\Pro\Agent-Graph\Long_Task_1
.\.venv\Scripts\python.exe -m engine.server
```

前端已运行 Vite 时刷新页面即可；未运行时在前端项目执行 `npm run dev:web`。不要同时启动两个占用同一端口的后端。

前端发起测试/审批前会检查 `/api/system/runtime` 的 `run_stream_protocol=2`，旧进程会收到明确的重启提示，不会悄悄发起新任务。OpenAPI 的 `info.x-run-stream-protocol` 也提供无敏感信息的启动版本标记；`npm run dev` 不再仅凭端口可访问就复用旧后端。Windows 启动器优先使用后端 `.venv`。

运行的创建、查询、审批与 SSE 使用相同的配置地址；写请求发生网络错误时不自动换到另一个后端重发。审批续跑时忽略同一旧审批的暂停通知，只有不同的未处理审批才重新显示待确认。执行动态标题中的“正文流式更新 N 次”表示收到真实正文分片的次数。

保存应用，进入测试区发送消息。执行动态显示节点、模型、工具、知识库检索、Skill 注入的实际进度；模型正文逐段显示。执行动态可折叠，正文与事件摘要分开显示。

## 协议与实现

- 继续使用 `POST /api/apps/{id}/runs`、`GET /api/runs/{id}` 和 `GET /api/runs/{id}/events`，不新增运行/会话创建协议。
- 新增事件：`model_started`、`model_finished`、`model_failed`、`answer_delta`、`answer_mode`、`tool_started`、`tool_finished`、`skill_applied`。知识库事件在真实检索处发出，不在节点结束后补造。
- 事件包含 `run_id`、`node`、`sequence`、`timestamp`。模型事件另含 `generation_id`，每次模型调用单独标识，避免多节点、多次调用或重试的文本串接。
- `answer_delta.delta` 仅为正文增量。约 100ms 或累积 256 字符合并一次，短回复在结束时刷新。不是整段返回后的打字动画。
- 同步推理/工具请求从事件循环移至工作线程；请求级 ContextVar 传递事件，不向工作流状态写入回调对象。
- Run 快照通过临时文件原子替换，避免 SSE 读取半个 JSON；取消标记不会被工作线程的旧副本覆盖。
- SSE 支持 `after` 和 `Last-Event-ID`，按事件序号续接。前端最多自动重连五次，之后显示手动重连按钮；重连不创建新 Run、不重新批准工具。

## 安全与真实状态

- 执行摘要是根据真实事件生成的简短说明，不是模型的原始思维链。不会把 `reasoning_content`、`<think>` 或 `<analysis>` 内容展示到正文中；必要的 provider reasoning 仅在内存中用于兼容工具调用协议。
- 工具参数流先完整组装，再解析与执行。高风险工具仍需批准；批准后继续原 Run，并防止并发审批重复执行。
- 等待批准时不继续发起兼容模式的模型生成，避免模型离线导致审批事件丢失。
- 流中断或缺少完成标记会报错并保留已收到的正文。不会把断流当作成功，也不会用本地模拟回答覆盖真实流式失败。
- 只有服务明确以 HTTP 400/422 拒绝流式参数，才尝试一次非流式请求，并在页面说明；流已经开始后不做这样的重试。原有非流式边侧服务仍使用其原协议。
- 停止是协作式的：不开始后续操作；正在进行的外部调用可能需要等它返回或超时。已完成的写入、发送等副作用不会撤销。
- 新执行动态只展示允许的摘要字段，不直接展开工具完整参数、密钥或私有 Skill 正文。原有运行详情/审计数据协议保持兼容。

## 验证

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_live_events.py -q -p no:cacheprovider
```

前端目录：

```powershell
node --test tests/run-stream.test.mjs
npm run build
```

新增测试使用离线模型分片与真实本地 Run/SSE 接口，覆盖正文/思考分离、工具参数组装、失败、取消、请求隔离、审批续跑及幂等、序号恢复和前端重连。真实远程服务能否流式返回还取决于所配置的模型供应商；没有用用户的远程模型额度进行验收。
