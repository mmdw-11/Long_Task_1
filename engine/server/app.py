"""FastAPI 服务：把 Orchestrator 的能力暴露为 REST 接口，供前端可视化编排。

提供的接口（前缀 /api）：

Agent 管理
- POST   /api/agents                 创建 agent
- GET    /api/agents                 列出所有 agent
- GET    /api/agents/{id}            查看单个 agent
- DELETE /api/agents/{id}            删除 agent
- POST   /api/agents/{id}/sub-agents 给 agent 添加子 agent（可多次）

连线管理
- POST   /api/connections            连接两个 agent（支持条件边）
- DELETE /api/connections            断开连线

图 / 执行
- GET    /api/graph                  导出可视化图结构（节点+边）
- POST   /api/graph/entry            设置入口 agent
- POST   /api/run                    执行编排并返回最终状态
- GET    /api/export                 导出完整编排 JSON
- POST   /api/import                 从 JSON 导入编排

运行方式：
    uvicorn engine.server.app:app --reload
或：
    python -m engine.server
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - 取决于运行环境
    raise ImportError(
        "启动 REST 服务需要安装 fastapi 与 uvicorn：pip install fastapi 'uvicorn[standard]'"
    ) from exc

from ..constants import END
from ..modules.agent_runtime import AgentRuntimeFactory
from ..modules.auth import AuthStore
from ..modules.context import ContextPolicy
from ..modules.context.todo import TodoManager
from ..modules.product_ops import (
    ApiKeyStore,
    ApplicationRecord,
    ApplicationStore,
    ProjectSnapshotService,
    ProductStatusService,
    ToolCatalogStore,
    ToolRecord,
)
from ..modules.security_ops import ApiAuditRecord, ApiAuditStore, utc_now
from ..modules.skills import (
    SkillEvolutionService,
    SkillRepository,
    SkillRetriever,
    SkillStatus,
    SkillTraceStore,
)
from ..modules.workflows import RunRecord, RunStore, WorkflowRecord, WorkflowStore
from ..modules.tool_runtime import ensure_builtin_tools
from ..orchestrator import NodeFactory, Orchestrator, _load_dotenv_for_context_policy


# ---------------------------------------------------------------------- #
# 请求体模型
# ---------------------------------------------------------------------- #
class CreateAgentReq(BaseModel):
    name: str
    sys_prompt: str = ""
    model: str = ""
    description: str = ""
    config: Dict[str, Any] = {}


class UpdateAgentReq(BaseModel):
    name: Optional[str] = None
    sys_prompt: Optional[str] = None
    model: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class AddSubAgentReq(BaseModel):
    # 二选一：挂载已有 child_id，或用 name 等新建
    child_id: Optional[str] = None
    name: Optional[str] = None
    sys_prompt: str = ""
    model: str = ""
    description: str = ""
    auto_connect: bool = True


class ConnectReq(BaseModel):
    source_id: str
    target_id: str  # 另一个 agent id，或字符串 "END"
    conditional: bool = False
    condition_key: Optional[str] = None
    path_map: Dict[str, str] = {}


class DisconnectReq(BaseModel):
    source_id: str
    target_id: str


class EntryReq(BaseModel):
    agent_id: str


class RunReq(BaseModel):
    input: Dict[str, Any] = {}
    recursion_limit: int = 50


class CreateRunReq(BaseModel):
    input: Dict[str, Any] = Field(default_factory=dict)
    recursion_limit: int = 50
    workflow_id: Optional[str] = None


class CancelRunReq(BaseModel):
    reason: str = ""


class ApprovalDecisionReq(BaseModel):
    reason: str = ""


class SaveWorkflowReq(BaseModel):
    name: str
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    graph: Optional[Dict[str, Any]] = None
    workflow_id: Optional[str] = None


class UpdateWorkflowReq(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None


class CreateSkillCandidateReq(BaseModel):
    run_id: str
    name: Optional[str] = None
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateSkillReq(BaseModel):
    name: str
    content: str
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SkillDecisionReq(BaseModel):
    approved_by: Optional[str] = None
    reason: str = ""


class SkillRolloutReq(BaseModel):
    percent: int = Field(100, ge=0, le=100)
    approved_by: str


class SkillSearchReq(BaseModel):
    query: str
    node: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 3


class CreateToolReq(BaseModel):
    name: str
    display_name: str
    description: str = ""
    category: str = "general"
    enabled: bool = True
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateToolReq(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    enabled: Optional[bool] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class CreateApiKeyReq(BaseModel):
    name: str
    scope: str = "workspace"


class UpdateApiKeyReq(BaseModel):
    enabled: bool


class CreateApplicationReq(BaseModel):
    name: str
    app_type: str = "agent"
    description: str = ""
    model: str = ""
    system_prompt: str = ""
    tool_ids: List[str] = Field(default_factory=list)
    skill_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateApplicationReq(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    tool_ids: Optional[List[str]] = None
    skill_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class RegisterReq(BaseModel):
    email: str
    name: str
    password: str


class LoginReq(BaseModel):
    email: str
    password: str


class ForgotPasswordReq(BaseModel):
    email: str


class ResetPasswordReq(BaseModel):
    token: str
    password: str


def _resolve_target(target: str) -> Any:
    """把接口传入的字符串目标解析为内部值（"END" -> END 哨兵）。"""
    return END if target == "END" else target


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_run_summary(record: RunRecord) -> Dict[str, Any]:
    """基于可见执行事件生成稳定、完整且可审计的运行总结。"""
    steps: List[Dict[str, Any]] = []
    for event in record.events:
        if event.get("type") != "node_end":
            continue
        output = event.get("output")
        if not output:
            update = dict(event.get("update") or {})
            messages = list(update.get("messages") or [])
            latest = messages[-1] if messages else {}
            output = latest.get("content") if isinstance(latest, dict) else None
        steps.append(
            {
                "order": len(steps) + 1,
                "agent": event.get("node"),
                "output": str(output or "")[:4000],
                "executor": event.get("executor"),
                "model": event.get("model"),
                "tool_calls": list(event.get("tool_calls") or []),
                "completed_at": event.get("timestamp"),
            }
        )
    final_text = ""
    messages = list(record.state.get("messages") or [])
    if messages and isinstance(messages[-1], dict):
        final_text = str(messages[-1].get("content") or "")
    if not final_text:
        final_text = str(record.state.get("input") or "")
    return {
        "title": "运行完成" if record.status == "succeeded" else "运行未成功完成",
        "status": record.status,
        "overview": (
            f"共执行 {len(steps)} 个 Agent 步骤，"
            f"产生 {len(record.events)} 条可见运行事件。"
        ),
        "steps": steps,
        "final_output": final_text,
        "errors": [record.error] if record.error else [],
    }


def create_app(
    orchestrator: Optional[Orchestrator] = None,
    *,
    context_policy: Optional[ContextPolicy] = None,
    context_policy_path: Optional[str] = None,
    context_ledger_root: Optional[str] = None,
    workflow_store: Optional[WorkflowStore] = None,
    run_store: Optional[RunStore] = None,
    skill_repository: Optional[SkillRepository] = None,
    skill_trace_store: Optional[SkillTraceStore] = None,
    node_factory: Optional[NodeFactory] = None,
    tool_catalog_store: Optional[ToolCatalogStore] = None,
    application_store: Optional[ApplicationStore] = None,
    api_key_store: Optional[ApiKeyStore] = None,
    api_audit_store: Optional[ApiAuditStore] = None,
    auth_store: Optional[AuthStore] = None,
    auth_required: bool = False,
) -> "FastAPI":
    """创建并返回 FastAPI 应用。可注入已有 Orchestrator，便于测试。"""
    orch = orchestrator or Orchestrator()
    policy = context_policy or _load_context_policy(context_policy_path)
    ledger_root = context_ledger_root or os.environ.get("CONTEXT_LEDGER_ROOT") or "runs/context"
    workflows = workflow_store or WorkflowStore(
        os.environ.get("WORKFLOW_STORE_ROOT") or "runs/workflows"
    )
    runs = run_store or RunStore(os.environ.get("RUN_STORE_ROOT") or "runs/executions")
    skills = skill_repository or SkillRepository(
        os.environ.get("SKILL_STORE_ROOT") or "runs/skills"
    )
    owns_tool_catalog = tool_catalog_store is None
    tools = tool_catalog_store or ToolCatalogStore(
        os.environ.get("TOOL_CATALOG_ROOT") or "runs/tool_catalog"
    )
    if owns_tool_catalog or os.environ.get("SEED_BUILTIN_TOOLS") == "1":
        ensure_builtin_tools(tools)
    applications = application_store or ApplicationStore(
        os.environ.get("APPLICATION_STORE_ROOT") or "runs/applications"
    )
    api_keys = api_key_store or ApiKeyStore(os.environ.get("API_KEY_STORE_ROOT") or "runs/api_keys")
    api_audit = api_audit_store or ApiAuditStore(
        os.environ.get("API_AUDIT_LOG_PATH") or "runs/audit/api_audit.jsonl"
    )
    auth = auth_store or AuthStore(os.environ.get("AUTH_DB_PATH") or "runs/auth/users.sqlite3")
    skill_traces = skill_trace_store or SkillTraceStore(
        os.environ.get("SKILL_TRACE_ROOT") or "runs/skill_traces"
    )
    skill_retriever = SkillRetriever(skills)
    skill_evolution = SkillEvolutionService(
        repository=skills,
        run_store=runs,
        trace_store=skill_traces,
    )
    runtime_factory = node_factory or AgentRuntimeFactory(tool_catalog_store=tools)
    admin_api_key = os.environ.get("ADMIN_API_KEY", "").strip()
    system_status = ProductStatusService(
        workflow_root=str(workflows.root_dir),
        run_root=str(runs.root_dir),
        skill_root=str(skills.root_dir),
        tool_root=str(tools.root_dir),
    )
    project_snapshot = ProjectSnapshotService(
        workflows=workflows,
        runs=runs,
        skills=skills,
        tools=tools,
        status_service=system_status,
    )
    if policy is not None:
        orch.set_context_policy(policy, ledger_root=ledger_root)
    orch.set_skill_retriever(skill_retriever)
    orch.set_skill_trace_store(skill_traces)
    app = FastAPI(title="Agent 编排服务", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\]):\d+",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def admin_guard_and_audit(request: Request, call_next):
        # CORS preflight does not carry a bearer token. Let CORSMiddleware
        # answer it before applying the API authentication policy.
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        auth_public = request.url.path in {
            "/api/auth/register", "/api/auth/login", "/api/auth/forgot-password", "/api/auth/reset-password"
        }
        authorization = request.headers.get("Authorization", "")
        bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        if not bearer and request.url.path.endswith("/events"):
            bearer = request.query_params.get("access_token", "")
        if not bearer:
            bearer = request.cookies.get("agentforge_session", "")
        user = auth.user_for_token(bearer)
        if auth_required and request.url.path.startswith("/api/") and not auth_public and user is None:
            return JSONResponse(status_code=401, content={"detail": "请先登录"})
        request.state.user = user
        # 仅对写接口启用最小保护；读接口保留无密钥可访问能力。
        requires_admin = (
            request.method.upper() in {"POST", "PUT", "DELETE"}
            and request.url.path.startswith("/api/")
            and not auth_public
        )
        actor = user.email if user else request.headers.get("X-Actor", "")
        provided_key = request.headers.get("X-Admin-Key", "")
        if requires_admin and admin_api_key and provided_key != admin_api_key:
            api_audit.append(
                ApiAuditRecord(
                    ts=utc_now(),
                    method=request.method.upper(),
                    path=request.url.path,
                    status_code=401,
                    actor=actor,
                    authorized=False,
                    detail="invalid admin key",
                )
            )
            return JSONResponse(status_code=401, content={"detail": "invalid admin key"})
        response = await call_next(request)
        if requires_admin:
            api_audit.append(
                ApiAuditRecord(
                    ts=utc_now(),
                    method=request.method.upper(),
                    path=request.url.path,
                    status_code=int(response.status_code),
                    actor=actor,
                    authorized=True,
                )
            )
        return response

    # ------------------------- 账户与会话 ------------------------- #
    def _set_session_cookie(response: Response, token: str) -> None:
        response.set_cookie(
            "agentforge_session",
            token,
            max_age=7 * 24 * 60 * 60,
            httponly=True,
            samesite="lax",
            secure=False,
            path="/",
        )

    @app.post("/api/auth/register")
    def register(req: RegisterReq, response: Response) -> Dict[str, Any]:
        try:
            user = auth.register(req.email, req.name, req.password)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        token = auth.create_session(user.id)
        _set_session_cookie(response, token)
        return {"token": token, "user": user.to_dict()}

    @app.post("/api/auth/login")
    def login(req: LoginReq, response: Response) -> Dict[str, Any]:
        user = auth.authenticate(req.email, req.password)
        if user is None:
            raise HTTPException(status_code=401, detail="邮箱或密码错误")
        token = auth.create_session(user.id)
        _set_session_cookie(response, token)
        return {"token": token, "user": user.to_dict()}

    @app.get("/api/auth/me")
    def current_user(request: Request) -> Dict[str, Any]:
        user = request.state.user
        if user is None:
            raise HTTPException(status_code=401, detail="请先登录")
        return {"user": user.to_dict()}

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response) -> Dict[str, Any]:
        authorization = request.headers.get("Authorization", "")
        token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        if not token:
            token = request.cookies.get("agentforge_session", "")
        if token:
            auth.revoke_session(token)
        response.delete_cookie("agentforge_session", path="/")
        return {"ok": True}

    @app.post("/api/auth/forgot-password")
    def forgot_password(req: ForgotPasswordReq) -> Dict[str, Any]:
        token = auth.create_password_reset(req.email)
        payload: Dict[str, Any] = {"message": "如果该邮箱已注册，重置说明将被发送"}
        # 本地开发没有邮件服务时返回一次性令牌；生产环境必须关闭并由邮件适配器投递。
        if token and os.environ.get("AUTH_EXPOSE_RESET_TOKEN", "1") == "1":
            payload["reset_token"] = token
        return payload

    @app.post("/api/auth/reset-password")
    def reset_password(req: ResetPasswordReq) -> Dict[str, Any]:
        try:
            updated = auth.reset_password(req.token, req.password)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not updated:
            raise HTTPException(status_code=400, detail="重置链接无效或已过期")
        return {"ok": True}

    def _apply_policy(target: Orchestrator) -> None:
        if policy is not None:
            target.set_context_policy(policy, ledger_root=ledger_root)
        target.set_skill_retriever(skill_retriever)
        target.set_skill_trace_store(skill_traces)

    def _orchestrator_for_run(workflow_id: Optional[str]) -> Orchestrator:
        if workflow_id:
            target = Orchestrator.from_dict(workflows.get(workflow_id).graph)
        else:
            target = Orchestrator.from_dict(orch.to_dict())
        _apply_policy(target)
        return target

    def _append_event(record: RunRecord, event: Dict[str, Any]) -> Dict[str, Any]:
        enriched = {
            **event,
            "sequence": len(record.events) + 1,
            "timestamp": _utc_now(),
        }
        record.events.append(enriched)
        runs.save(record)
        return enriched

    def _task_text(payload: Dict[str, Any]) -> str:
        for key in ("input", "task", "goal", "query"):
            value = payload.get(key)
            if value not in (None, ""):
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _initial_plan(payload: Dict[str, Any], target: Orchestrator) -> List[str]:
        explicit = payload.get("current_plan") or payload.get("plan")
        if isinstance(explicit, list) and explicit:
            return [str(item) for item in explicit if str(item).strip()]
        graph = target.to_dict()
        agents = [item for item in graph.get("agents", []) if isinstance(item, dict)]
        if agents:
            return [
                f"{agent.get('name') or agent.get('id')}：{agent.get('description') or '完成该节点负责的任务'}"
                for agent in agents
            ]
        task = _task_text(payload)
        return [
            f"理解任务目标：{task[:80]}",
            "拆解关键步骤并收集必要上下文",
            "执行任务并记录证据、工具结果与中间产物",
            "核对输出完整性并生成最终总结",
        ]

    def _seed_todo_events(record: RunRecord, target: Orchestrator) -> None:
        plan = _initial_plan(record.input, target)
        record.input = {**record.input, "current_plan": plan, "original_goal": _task_text(record.input)}
        record.metadata = {**record.metadata, "todos": []}
        if policy is not None and target._context_ledger is not None:  # noqa: SLF001 - service-level integration hook
            ledger = target._context_ledger.load_or_create(record.id, {**record.input, "run_id": record.id})  # noqa: SLF001
            TodoManager(ledger).set_todos(plan, source="planner", reason="initial run planning")
            target._context_ledger.save(ledger)  # noqa: SLF001
            todos = [item.to_dict() for item in ledger.todo_items]
        else:
            todos = [
                {
                    "id": f"todo-{idx}",
                    "content": item,
                    "status": "in_progress" if idx == 1 else "pending",
                    "source": "planner",
                }
                for idx, item in enumerate(plan, 1)
            ]
        record.metadata["todos"] = todos
        _append_event(
            record,
            {
                "type": "plan_created",
                "message": "已根据任务和工作流生成执行 TODO",
                "plan": plan,
                "todos": todos,
            },
        )

    def _advance_todo(record: RunRecord, *, node: str, output: str = "") -> None:
        todos = list(record.metadata.get("todos") or [])
        if not todos:
            return
        active_idx = next((idx for idx, item in enumerate(todos) if item.get("status") == "in_progress"), -1)
        if active_idx < 0:
            active_idx = next((idx for idx, item in enumerate(todos) if item.get("status") == "pending"), -1)
            if active_idx >= 0:
                todos[active_idx]["status"] = "in_progress"
        if active_idx < 0:
            return
        todos[active_idx]["status"] = "completed"
        todos[active_idx]["evidence"] = output[:500] or f"{node} completed"
        next_idx = next((idx for idx, item in enumerate(todos) if item.get("status") == "pending"), -1)
        if next_idx >= 0:
            todos[next_idx]["status"] = "in_progress"
        record.metadata["todos"] = todos
        _append_event(
            record,
            {
                "type": "todo_updated",
                "node": node,
                "message": f"{node} 完成后已更新 TODO 状态",
                "todos": todos,
                "active_todo_id": todos[next_idx]["id"] if next_idx >= 0 else "",
            },
        )

    async def _execute_run(record: RunRecord) -> None:
        started_clock = time.perf_counter()
        try:
            record.status = "running"
            record.metadata = {
                **record.metadata,
                "started_at": _utc_now(),
                "active_agent": None,
                "summary": None,
            }
            runs.save(record)
            target = _orchestrator_for_run(record.workflow_id)
            _seed_todo_events(record, target)
            compiled = target.build_graph(
                node_factory=runtime_factory,
                recursion_limit=record.recursion_limit,
            )
            run_input = {**record.input, "run_id": record.id}
            async for event in compiled.astream(
                run_input,
                record.recursion_limit,
                run_id=record.id,
            ):
                if event.get("type") == "node_start":
                    record.metadata["active_agent"] = event.get("node")
                    event["message"] = f"进入 {event.get('node')}，开始处理当前步骤"
                elif event.get("type") == "node_end":
                    update = dict(event.get("update") or {})
                    messages = list(update.get("messages") or [])
                    latest_message = messages[-1] if messages else {}
                    result = dict(latest_message.get("result") or {}) if isinstance(latest_message, dict) else {}
                    event["output"] = latest_message.get("content") if isinstance(latest_message, dict) else None
                    event["executor"] = result.get("executor")
                    event["model"] = result.get("model")
                    event["tool_calls"] = list(result.get("metadata", {}).get("tool_calls") or [])
                    event["message"] = f"{event.get('node')} 已完成当前步骤"
                elif event.get("type") == "route":
                    event["message"] = f"{event.get('node')} 完成路由，下一步：{', '.join(event.get('targets') or ['结束'])}"
                # 长任务取消采用协作式检查，避免强杀执行线程导致状态文件损坏。
                latest = runs.get(record.id)
                if latest.status == "cancel_requested":
                    record.status = "canceled"
                    record.canceled_at = latest.canceled_at or _utc_now()
                    record.finished_at = record.canceled_at
                    record.metadata = latest.metadata
                    runs.save(record)
                    return
                _append_event(record, event)
                if event.get("type") == "node_end":
                    for tool_call in event.get("tool_calls") or []:
                        _append_event(
                            record,
                            {
                                "type": (
                                    "approval_required"
                                    if tool_call.get("status") == "approval_required"
                                    else "tool_result"
                                ),
                                "node": event.get("node"),
                                "tool_call": tool_call,
                                "message": (
                                    f"{event.get('node')} 请求审批工具 {tool_call.get('display_name') or tool_call.get('name')}"
                                    if tool_call.get("status") == "approval_required"
                                    else f"{event.get('node')} 已调用工具 {tool_call.get('display_name') or tool_call.get('name')}"
                                ),
                            },
                        )
                    _advance_todo(record, node=str(event.get("node") or ""), output=str(event.get("output") or ""))
                if event.get("type") == "final":
                    record.state = dict(event.get("state") or {})
                runs.save(record)
            record.status = "succeeded"
            record.finished_at = _utc_now()
            record.metadata = {
                **record.metadata,
                "active_agent": None,
                "duration_ms": round((time.perf_counter() - started_clock) * 1000, 2),
                "summary": _build_run_summary(record),
            }
            runs.save(record)
        except Exception as e:  # noqa: BLE001 - API persists failures for polling
            record.status = "failed"
            record.error = str(e)
            record.finished_at = _utc_now()
            record.metadata = {
                **record.metadata,
                "active_agent": None,
                "duration_ms": round((time.perf_counter() - started_clock) * 1000, 2),
                "summary": _build_run_summary(record),
            }
            runs.save(record)

    # ------------------------- Agent 管理 ------------------------- #
    @app.post("/api/agents")
    def create_agent(req: CreateAgentReq) -> Dict[str, Any]:
        try:
            aid = orch.create_agent(
                name=req.name,
                sys_prompt=req.sys_prompt,
                model=req.model,
                description=req.description,
                config=req.config,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return orch.get_agent(aid).to_dict()

    @app.get("/api/agents")
    def list_agents() -> List[Dict[str, Any]]:
        return [a.to_dict() for a in orch.list_agents()]

    @app.get("/api/system/status")
    def get_system_status() -> Dict[str, Any]:
        snapshot = system_status.snapshot()
        snapshot["security"] = {
            "admin_key_enabled": bool(admin_api_key),
            "audit_log_path": str(api_audit.path),
        }
        host = os.environ.get("AGENTFORGE_PUBLIC_BASE_URL") or "http://127.0.0.1:8000"
        snapshot["api_access"] = {
            "openai_compatible_base_url": f"{host.rstrip('/')}/compatible-mode/v1",
            "anthropic_base_url": f"{host.rstrip('/')}/apps/anthropic",
            "workspace": os.environ.get("AGENTFORGE_WORKSPACE_NAME") or "默认业务空间",
            "api_key_count": len(api_keys.list()),
            "application_count": len(applications.list()),
        }
        return snapshot

    @app.get("/api/system/audit-logs")
    def get_api_audit_logs(limit: int = 100) -> Dict[str, Any]:
        bounded_limit = max(1, min(limit, 500))
        return {"records": [item.to_dict() for item in api_audit.list(limit=bounded_limit)]}

    @app.get("/api/agents/{agent_id}")
    def get_agent(agent_id: str) -> Dict[str, Any]:
        try:
            return orch.get_agent(agent_id).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.put("/api/agents/{agent_id}")
    def update_agent(agent_id: str, req: UpdateAgentReq) -> Dict[str, Any]:
        try:
            return orch.update_agent(
                agent_id,
                name=req.name,
                sys_prompt=req.sys_prompt,
                model=req.model,
                description=req.description,
                config=req.config,
            ).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/agents/{agent_id}")
    def delete_agent(agent_id: str) -> Dict[str, Any]:
        try:
            orch.remove_agent(agent_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"ok": True}

    @app.post("/api/agents/{agent_id}/sub-agents")
    def add_sub_agent(agent_id: str, req: AddSubAgentReq) -> Dict[str, Any]:
        try:
            child_id = orch.add_sub_agent(
                parent_id=agent_id,
                child_id=req.child_id,
                name=req.name,
                sys_prompt=req.sys_prompt,
                model=req.model,
                description=req.description,
                auto_connect=req.auto_connect,
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return orch.get_agent(child_id).to_dict()

    # ------------------------- 连线管理 ------------------------- #
    @app.post("/api/connections")
    def connect(req: ConnectReq) -> Dict[str, Any]:
        try:
            if req.conditional:
                if not req.condition_key or not req.path_map:
                    raise ValueError("条件边需要提供 condition_key 与 path_map")
                path_map = {k: _resolve_target(v) for k, v in req.path_map.items()}
                orch.connect_conditional(req.source_id, req.condition_key, path_map)
            else:
                orch.connect(req.source_id, _resolve_target(req.target_id))
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"ok": True}

    @app.delete("/api/connections")
    def disconnect(req: DisconnectReq) -> Dict[str, Any]:
        orch.disconnect(req.source_id, _resolve_target(req.target_id))
        return {"ok": True}

    # ------------------------- 图 / 执行 ------------------------- #
    @app.post("/api/graph/entry")
    def set_entry(req: EntryReq) -> Dict[str, Any]:
        try:
            orch.set_entry(req.agent_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"ok": True}

    @app.get("/api/graph")
    def get_graph() -> Dict[str, Any]:
        try:
            compiled = orch.build_graph()
            return compiled.to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/run")
    async def run(req: RunReq) -> Dict[str, Any]:
        try:
            compiled = orch.build_graph(
                node_factory=runtime_factory,
                recursion_limit=req.recursion_limit,
            )
            state = await compiled.ainvoke(req.input, req.recursion_limit)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"state": state}

    @app.post("/api/runs")
    async def create_run(
        req: CreateRunReq,
        background_tasks: BackgroundTasks,
    ) -> Dict[str, Any]:
        try:
            if req.workflow_id:
                workflows.get(req.workflow_id)
            record = runs.create(
                input=req.input,
                recursion_limit=req.recursion_limit,
                workflow_id=req.workflow_id,
            )
            record.status = "queued"
            runs.save(record)
            background_tasks.add_task(_execute_run, record)
            return record.to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/runs")
    def list_runs(workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in runs.list(workflow_id=workflow_id)]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> Dict[str, Any]:
        try:
            return runs.get(run_id).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/runs/{run_id}/events")
    async def stream_run_events(run_id: str, after: int = 0):
        """用 SSE 推送持久化运行事件；断线后可通过 after 继续。"""
        try:
            runs.get(run_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

        async def _event_stream():
            cursor = max(0, after)
            idle_ticks = 0
            while True:
                record = runs.get(run_id)
                while cursor < len(record.events):
                    event = record.events[cursor]
                    cursor += 1
                    yield f"id: {cursor}\nevent: run_event\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    idle_ticks = 0
                if record.status in {"succeeded", "failed", "canceled"}:
                    completed = {
                        "type": "run_completed",
                        "status": record.status,
                        "finished_at": record.finished_at,
                        "duration_ms": record.metadata.get("duration_ms"),
                        "summary": record.metadata.get("summary"),
                    }
                    yield f"event: run_completed\ndata: {json.dumps(completed, ensure_ascii=False)}\n\n"
                    break
                idle_ticks += 1
                if idle_ticks % 15 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str, req: CancelRunReq) -> Dict[str, Any]:
        try:
            return runs.mark_cancel_requested(run_id, reason=req.reason).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/api/runs/{run_id}/approvals/{sequence}/approve")
    def approve_tool_call(run_id: str, sequence: int, req: ApprovalDecisionReq) -> Dict[str, Any]:
        return _record_approval_decision(run_id, sequence, approved=True, reason=req.reason)

    @app.post("/api/runs/{run_id}/approvals/{sequence}/reject")
    def reject_tool_call(run_id: str, sequence: int, req: ApprovalDecisionReq) -> Dict[str, Any]:
        return _record_approval_decision(run_id, sequence, approved=False, reason=req.reason)

    def _record_approval_decision(run_id: str, sequence: int, *, approved: bool, reason: str = "") -> Dict[str, Any]:
        try:
            record = runs.get(run_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        target = next(
            (
                event
                for event in record.events
                if int(event.get("sequence") or 0) == sequence and event.get("type") == "approval_required"
            ),
            None,
        )
        if target is None:
            raise HTTPException(status_code=404, detail="approval event not found")
        decisions = dict(record.metadata.get("approval_decisions") or {})
        decisions[str(sequence)] = {
            "approved": approved,
            "reason": reason,
            "decided_at": _utc_now(),
        }
        record.metadata["approval_decisions"] = decisions
        _append_event(
            record,
            {
                "type": "approval_decision",
                "approval_sequence": sequence,
                "approved": approved,
                "reason": reason,
                "tool_call": target.get("tool_call"),
                "message": "用户已批准工具请求" if approved else "用户已拒绝工具请求",
            },
        )
        runs.save(record)
        return record.to_dict()

    @app.post("/api/runs/{run_id}/retry")
    async def retry_run(run_id: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        try:
            record = runs.retry(run_id)
            record.status = "queued"
            runs.save(record)
            background_tasks.add_task(_execute_run, record)
            return record.to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/system/metrics")
    def get_system_metrics() -> Dict[str, Any]:
        return {
            "runs": runs.metrics(),
            "applications": {"total": len(applications.list())},
            "workflows": {"total": len(workflows.list())},
            "skills": {"total": len(skills.list())},
            "tools": {"total": len(tools.list())},
        }

    @app.get("/api/system/export")
    def export_system_snapshot(
        include_runs: bool = True,
        include_skill_content: bool = True,
    ) -> Dict[str, Any]:
        return project_snapshot.export(
            include_runs=include_runs,
            include_skill_content=include_skill_content,
        )

    @app.get("/api/export")
    def export() -> Dict[str, Any]:
        return orch.to_dict()

    @app.post("/api/import")
    def import_(data: Dict[str, Any]) -> Dict[str, Any]:
        # nonlocal 重绑定会更新所有闭包共享的同一变量，后续接口即读取新实例。
        nonlocal orch
        orch = Orchestrator.from_dict(data)
        _apply_policy(orch)
        return {"ok": True, "agents": len(data.get("agents", []))}

    # ------------------------- 应用中心 ------------------------- #
    @app.get("/api/apps")
    def list_applications() -> List[Dict[str, Any]]:
        return [item.to_dict() for item in applications.list()]

    @app.post("/api/apps")
    def create_application(req: CreateApplicationReq) -> Dict[str, Any]:
        try:
            app_record = applications.create(
                name=req.name,
                app_type=req.app_type,
                description=req.description,
                model=req.model,
                system_prompt=req.system_prompt,
                tool_ids=req.tool_ids,
                skill_ids=req.skill_ids,
                metadata=req.metadata,
            )
            draft = Orchestrator()
            entry_id = draft.create_agent(
                name=req.name,
                sys_prompt=req.system_prompt,
                model=req.model,
                description=req.description or "负责应用入口任务理解、工具调用和最终答复。",
                config={"tool_ids": req.tool_ids, "skill_ids": req.skill_ids},
            )
            draft.set_entry(entry_id)
            workflow = workflows.create(
                name=req.name,
                description=req.description,
                tags=[req.app_type, "application"],
                metadata={"application_id": app_record.id, **req.metadata},
                graph=draft.to_dict(),
            )
            app_record.workflow_id = workflow.id
            app_record.entry_agent_id = entry_id
            app_record.metadata = {**app_record.metadata, "workflow_version": workflow.version}
            applications.save(app_record)
            return app_record.to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/apps/{app_id}")
    def get_application(app_id: str) -> Dict[str, Any]:
        try:
            app_record = applications.get(app_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        payload = app_record.to_dict()
        if app_record.workflow_id:
            try:
                payload["workflow"] = workflows.get(app_record.workflow_id).to_dict()
            except KeyError:
                payload["workflow_missing"] = True
        return payload

    @app.put("/api/apps/{app_id}")
    def update_application(app_id: str, req: UpdateApplicationReq) -> Dict[str, Any]:
        try:
            current = applications.get(app_id)
            updated = ApplicationRecord.from_dict(
                {
                    **current.to_dict(),
                    "name": req.name if req.name is not None else current.name,
                    "description": req.description if req.description is not None else current.description,
                    "status": req.status if req.status is not None else current.status,
                    "model": req.model if req.model is not None else current.model,
                    "system_prompt": req.system_prompt if req.system_prompt is not None else current.system_prompt,
                    "tool_ids": req.tool_ids if req.tool_ids is not None else current.tool_ids,
                    "skill_ids": req.skill_ids if req.skill_ids is not None else current.skill_ids,
                    "metadata": req.metadata if req.metadata is not None else current.metadata,
                }
            )
            if updated.workflow_id:
                workflow = workflows.get(updated.workflow_id)
                graph = Orchestrator.from_dict(workflow.graph)
                if updated.entry_agent_id:
                    graph.update_agent(
                        updated.entry_agent_id,
                        name=updated.name,
                        sys_prompt=updated.system_prompt,
                        model=updated.model,
                        description=updated.description,
                        config={"tool_ids": updated.tool_ids, "skill_ids": updated.skill_ids},
                    )
                workflows.save(
                    WorkflowRecord.from_dict(
                        {
                            **workflow.to_dict(),
                            "name": updated.name,
                            "description": updated.description,
                            "graph": graph.to_dict(),
                        }
                    )
                )
            return applications.save(updated).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/apps/{app_id}/publish")
    def publish_application(app_id: str) -> Dict[str, Any]:
        try:
            app_record = applications.get(app_id)
            app_record.status = "published"
            return applications.save(app_record).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.delete("/api/apps/{app_id}")
    def delete_application(app_id: str) -> Dict[str, Any]:
        try:
            applications.delete(app_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"ok": True}

    # ------------------------- 工作流持久化 ------------------------- #
    @app.post("/api/workflows")
    def save_workflow(req: SaveWorkflowReq) -> Dict[str, Any]:
        try:
            record = workflows.create(
                workflow_id=req.workflow_id,
                name=req.name,
                description=req.description,
                tags=req.tags,
                metadata=req.metadata,
                graph=req.graph or orch.to_dict(),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return record.to_dict()

    @app.get("/api/workflows")
    def list_workflows() -> List[Dict[str, Any]]:
        return [item.to_dict() for item in workflows.list()]

    @app.get("/api/workflows/{workflow_id}")
    def get_workflow(workflow_id: str) -> Dict[str, Any]:
        try:
            return workflows.get(workflow_id).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/workflows/{workflow_id}/versions")
    def list_workflow_versions(workflow_id: str) -> List[Dict[str, Any]]:
        try:
            return [item.to_dict() for item in workflows.list_versions(workflow_id)]
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/workflows/{workflow_id}/versions/{version}")
    def get_workflow_version(workflow_id: str, version: int) -> Dict[str, Any]:
        try:
            return workflows.get_version(workflow_id, version).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.put("/api/workflows/{workflow_id}")
    def update_workflow(workflow_id: str, req: UpdateWorkflowReq) -> Dict[str, Any]:
        try:
            current = workflows.get(workflow_id)
            updated = WorkflowRecord.from_dict(
                {
                    **current.to_dict(),
                    "name": req.name if req.name is not None else current.name,
                    "description": (
                        req.description
                        if req.description is not None
                        else current.description
                    ),
                    "tags": req.tags if req.tags is not None else current.tags,
                    "metadata": (
                        req.metadata if req.metadata is not None else current.metadata
                    ),
                    "graph": req.graph if req.graph is not None else current.graph,
                }
            )
            return workflows.save(updated).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/workflows/{workflow_id}/load")
    def load_workflow(workflow_id: str) -> Dict[str, Any]:
        nonlocal orch
        try:
            record = workflows.get(workflow_id)
            orch = Orchestrator.from_dict(record.graph)
            _apply_policy(orch)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"ok": True, "workflow": record.to_dict(), "graph": orch.to_dict()}

    @app.post("/api/workflows/{workflow_id}/rollback/{version}")
    def rollback_workflow(workflow_id: str, version: int) -> Dict[str, Any]:
        try:
            return workflows.rollback(workflow_id, version).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/workflows/{workflow_id}")
    def delete_workflow(workflow_id: str) -> Dict[str, Any]:
        try:
            workflows.delete(workflow_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"ok": True}

    # ------------------------- 技能生命周期 ------------------------- #
    @app.post("/api/skills")
    def create_skill(req: CreateSkillReq) -> Dict[str, Any]:
        try:
            return skills.create(
                name=req.name,
                content=req.content,
                description=req.description,
                tags=req.tags,
                metadata=req.metadata,
            ).to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/skills")
    def list_skills(status: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            return [item.to_dict() for item in skills.list(status=status)]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/search")
    def search_skills(req: SkillSearchReq) -> Dict[str, Any]:
        matches = skill_retriever.retrieve(
            req.query,
            node=req.node,
            metadata=req.metadata,
            top_k=req.top_k,
        )
        return {"matches": [item.to_dict() for item in matches]}

    @app.get("/api/skills/{skill_id}")
    def get_skill(skill_id: str) -> Dict[str, Any]:
        try:
            return skills.get(skill_id).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/skills/{skill_id}/versions")
    def list_skill_versions(skill_id: str) -> List[Dict[str, Any]]:
        try:
            return [item.to_dict() for item in skills.list_versions(skill_id)]
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/skills/{skill_id}/versions/{version}")
    def get_skill_version(skill_id: str, version: int) -> Dict[str, Any]:
        try:
            return skills.get_version(skill_id, version).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/candidates/from-run")
    def create_skill_candidate(req: CreateSkillCandidateReq) -> Dict[str, Any]:
        try:
            return skill_evolution.create_candidate_from_run(
                req.run_id,
                name=req.name,
                description=req.description,
                tags=req.tags,
                metadata=req.metadata,
            ).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/{skill_id}/validate")
    def validate_skill(skill_id: str) -> Dict[str, Any]:
        try:
            return skill_evolution.validate(skill_id).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/{skill_id}/publish")
    def publish_skill(skill_id: str, req: SkillDecisionReq) -> Dict[str, Any]:
        try:
            return skill_evolution.publish(
                skill_id,
                approved_by=req.approved_by or "",
            ).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/{skill_id}/reject")
    def reject_skill(skill_id: str, req: SkillDecisionReq) -> Dict[str, Any]:
        try:
            return skill_evolution.reject(skill_id, reason=req.reason).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/{skill_id}/retire")
    def retire_skill(skill_id: str, req: SkillDecisionReq) -> Dict[str, Any]:
        try:
            return skill_evolution.retire(skill_id, reason=req.reason).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/{skill_id}/rollback/{version}")
    def rollback_skill(skill_id: str, version: int, req: SkillDecisionReq) -> Dict[str, Any]:
        try:
            return skills.rollback(
                skill_id,
                version,
                approved_by=req.approved_by or "",
                reason=req.reason,
            ).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/{skill_id}/rollout")
    def set_skill_rollout(skill_id: str, req: SkillRolloutReq) -> Dict[str, Any]:
        try:
            return skills.set_rollout(
                skill_id,
                req.percent,
                approved_by=req.approved_by,
            ).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/runs/{run_id}/skill-traces")
    def get_skill_traces(run_id: str) -> Dict[str, Any]:
        return {"events": [item.to_dict() for item in skill_traces.list(run_id)]}

    # ------------------------- API Key 管理 ------------------------- #
    @app.get("/api/api-keys")
    def list_api_keys() -> List[Dict[str, Any]]:
        return [item.to_dict() for item in api_keys.list()]

    @app.post("/api/api-keys")
    def create_api_key(req: CreateApiKeyReq, request: Request) -> Dict[str, Any]:
        user = getattr(request.state, "user", None)
        created_by = user.email if user else request.headers.get("X-Actor", "")
        try:
            record, secret = api_keys.create(name=req.name, scope=req.scope, created_by=created_by)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return record.to_dict(include_secret=secret)

    @app.put("/api/api-keys/{key_id}")
    def update_api_key(key_id: str, req: UpdateApiKeyReq) -> Dict[str, Any]:
        try:
            return api_keys.update_enabled(key_id, req.enabled).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.delete("/api/api-keys/{key_id}")
    def delete_api_key(key_id: str) -> Dict[str, Any]:
        try:
            api_keys.delete(key_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"ok": True}

    # ------------------------- 工具目录 ------------------------- #
    @app.post("/api/tools")
    def create_tool(req: CreateToolReq) -> Dict[str, Any]:
        try:
            return tools.create(
                name=req.name,
                display_name=req.display_name,
                description=req.description,
                category=req.category,
                enabled=req.enabled,
                tags=req.tags,
                metadata=req.metadata,
            ).to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/tools")
    def list_tools(enabled: Optional[bool] = None) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in tools.list(enabled=enabled)]

    @app.get("/api/tools/{tool_id}")
    def get_tool(tool_id: str) -> Dict[str, Any]:
        try:
            return tools.get(tool_id).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.put("/api/tools/{tool_id}")
    def update_tool(tool_id: str, req: UpdateToolReq) -> Dict[str, Any]:
        try:
            current = tools.get(tool_id)
            updated = ToolRecord.from_dict(
                {
                    **current.to_dict(),
                    "display_name": req.display_name if req.display_name is not None else current.display_name,
                    "description": req.description if req.description is not None else current.description,
                    "category": req.category if req.category is not None else current.category,
                    "enabled": req.enabled if req.enabled is not None else current.enabled,
                    "tags": req.tags if req.tags is not None else current.tags,
                    "metadata": req.metadata if req.metadata is not None else current.metadata,
                }
            )
            return tools.save(updated).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/tools/{tool_id}")
    def delete_tool(tool_id: str) -> Dict[str, Any]:
        try:
            tools.delete(tool_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"ok": True}

    return app


def _load_context_policy(context_policy_path: Optional[str]) -> Optional[ContextPolicy]:
    _load_dotenv_for_context_policy()
    path = context_policy_path or os.environ.get("CONTEXT_POLICY_PATH")
    if not path:
        return None
    return ContextPolicy.from_file(path)


# 便于 `uvicorn engine.server.app:app` 直接启动。
_load_dotenv_for_context_policy()
_default_context_policy_path = os.environ.get("CONTEXT_POLICY_PATH")
if not _default_context_policy_path and os.path.exists("configs/context_policy.yaml"):
    _default_context_policy_path = "configs/context_policy.yaml"
app = create_app(
    context_policy_path=_default_context_policy_path,
    auth_required=os.environ.get("AUTH_REQUIRED", "1") == "1",
)
