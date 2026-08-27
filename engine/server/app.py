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
import urllib.error
import urllib.request
import zipfile
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
    ConsoleResourceStore,
    MemoryBankStore,
    ProjectSnapshotService,
    ProductStatusService,
    ToolCatalogStore,
    ToolRecord,
)
from ..modules.model_connections import MODEL_PRESETS, ModelConnection, ModelConnectionStore
from ..modules.external_imports import discover_mcp_tools, openapi_operations, parse_openapi, read_remote_document, read_skill_git, read_skill_zip, validate_remote_url
from ..modules.security_ops import ApiAuditRecord, ApiAuditStore, utc_now
from ..modules.skills import (
    SkillEvolutionService,
    SkillRepository,
    SkillRetriever,
    SkillStatus,
    SkillTraceStore,
)
from ..modules.workflows import RunRecord, RunStore, WorkflowRecord, WorkflowStore
from ..modules.tool_runtime import ToolRuntime, ensure_builtin_tools
from ..modules.workflow_runtime import WorkflowNodeRuntimeFactory
from ..orchestrator import NodeFactory, Orchestrator, _load_dotenv_for_context_policy


BUILTIN_SKILLS = [
    {"slug":"email-writer","name":"商务邮件撰写","category":"通用办公","description":"根据收件人、目的和语气起草清晰、可发送的商务邮件。","content":"# 商务邮件撰写\n\n先确认收件人、主题、目的和语气；给出结构化邮件草稿。发送前必须请求用户确认。"},
    {"slug":"research-report","name":"研究报告","category":"内容创意","description":"把研究主题拆解为目标、证据、结论与待验证项，避免虚构来源。","content":"# 研究报告\n\n先列出研究问题和证据需求，输出结论时标识事实、推断和待核验项。"},
    {"slug":"travel-planner","name":"旅行计划","category":"通用办公","description":"生成兼顾时间、预算、天气与交通的行程方案。","content":"# 旅行计划\n\n确认目的地、日期、预算、同行人和偏好；涉及实时信息时建议调用已授权工具。"},
    {"slug":"meeting-summary","name":"会议纪要","category":"通用办公","description":"将会议材料整理为结论、行动项、负责人和截止时间。","content":"# 会议纪要\n\n以结论、行动项、负责人、截止时间四部分输出；缺失信息明确标记待补充。"},
    {"slug":"web-design","name":"网页设计","category":"代码开发","description":"把用户需求转为信息架构、界面层级和可实施的前端建议。","content":"# 网页设计\n\n先给出页面目标、用户路径和组件清单，再输出可实施的视觉与交互建议。"},
    {"slug":"data-analysis","name":"数据分析","category":"金融","description":"帮助解释指标、识别异常并给出可复现的分析路径。","content":"# 数据分析\n\n明确数据范围与口径，区分计算结果和业务推断，给出复核步骤。"},
]


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


class TestToolReq(BaseModel):
    """从控制台验证工具适配器；高风险工具只返回审批请求。"""

    task: str = "请执行工具连通性测试"


class CreateApiKeyReq(BaseModel):
    name: str
    scope: str = "workspace"


class UpdateApiKeyReq(BaseModel):
    enabled: bool


class CreateApplicationReq(BaseModel):
    name: str = Field(max_length=50)
    app_type: str = "agent"
    description: str = ""
    model: str = ""
    system_prompt: str = ""
    avatar_url: str = ""
    tool_ids: List[str] = Field(default_factory=list)
    skill_ids: List[str] = Field(default_factory=list)
    knowledge_base_ids: List[str] = Field(default_factory=list)
    memory_bank_ids: List[str] = Field(default_factory=list)
    prompt_variables: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateApplicationReq(BaseModel):
    name: Optional[str] = Field(default=None, max_length=50)
    description: Optional[str] = None
    status: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    avatar_url: Optional[str] = None
    tool_ids: Optional[List[str]] = None
    skill_ids: Optional[List[str]] = None
    knowledge_base_ids: Optional[List[str]] = None
    memory_bank_ids: Optional[List[str]] = None
    prompt_variables: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class CreateApplicationRunReq(BaseModel):
    input: Dict[str, Any] = Field(default_factory=dict)
    recursion_limit: int = 50


class CreateMemoryBankReq(BaseModel):
    """创建控制台可挂载的记忆库资源。"""

    name: str
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateConsoleResourceReq(BaseModel):
    """创建组件、知识库、数据连接或治理资源。"""

    name: str
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    memory_bank_store: Optional[MemoryBankStore] = None,
    console_resource_store: Optional[ConsoleResourceStore] = None,
    api_key_store: Optional[ApiKeyStore] = None,
    api_audit_store: Optional[ApiAuditStore] = None,
    auth_store: Optional[AuthStore] = None,
    auth_required: bool = False,
    model_connection_store: Optional[ModelConnectionStore] = None,
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
    model_connections = model_connection_store or ModelConnectionStore(os.environ.get("MODEL_CONNECTION_ROOT") or "runs/model_connections")
    memory_banks = memory_bank_store or MemoryBankStore(
        os.environ.get("MEMORY_BANK_STORE_ROOT") or "runs/memory_banks"
    )
    console_resources = console_resource_store or ConsoleResourceStore(
        os.environ.get("CONSOLE_RESOURCE_STORE_ROOT") or "runs/console_resources"
    )
    api_keys = api_key_store or ApiKeyStore(os.environ.get("API_KEY_STORE_ROOT") or "runs/api_keys")
    api_audit = api_audit_store or ApiAuditStore(
        os.environ.get("API_AUDIT_LOG_PATH") or "runs/audit/api_audit.jsonl"
    )
    auth = auth_store or AuthStore(os.environ.get("AUTH_DB_PATH") or "runs/auth/users.sqlite3")
    # 内置 Skill 是平台可信只读能力，启动时幂等预置并直接发布。
    for template in BUILTIN_SKILLS:
        existing = next((item for item in skills.list() if item.metadata.get("market_slug") == template["slug"]), None)
        if existing is None:
            skills.create(name=template["name"],content=template["content"],description=template["description"],tags=[template["category"],"market","builtin"],status=SkillStatus.PUBLISHED,visibility="builtin",source_type="builtin",validation_status="passed",metadata={"market_slug":template["slug"],"source":"builtin","category":template["category"],"migration_version":1})
        elif existing.visibility != "builtin" or existing.status != SkillStatus.PUBLISHED:
            existing.status=SkillStatus.PUBLISHED;existing.owner_user_id=None;existing.visibility="builtin";existing.source_type="builtin";existing.validation_status="passed";existing.metadata={**existing.metadata,"source":"builtin","category":template["category"],"migration_version":1};skills.save(existing)
    # 旧资源没有所有者；在存在账号时一次性归属最早注册用户。
    legacy_owner = auth.first_user()
    legacy_owner_id = (legacy_owner.id if legacy_owner is not None else "") if auth_required else "local-user"
    migration_marker = applications.root_dir / ".ownership-v1"
    if legacy_owner_id:
        for record in applications.list():
            if not record.owner_user_id:
                record.owner_user_id=legacy_owner_id;applications.save(record)
        for record in skills.list():
            if record.visibility != "builtin" and not record.owner_user_id:
                record.owner_user_id=legacy_owner_id;record.visibility="private";skills.save(record,versioned=False)
        migration_marker.write_text(legacy_owner_id,encoding="utf-8")
    skill_traces = skill_trace_store or SkillTraceStore(
        os.environ.get("SKILL_TRACE_ROOT") or "runs/skill_traces"
    )
    skill_retriever = SkillRetriever(skills)
    skill_evolution = SkillEvolutionService(
        repository=skills,
        run_store=runs,
        trace_store=skill_traces,
    )
    runtime_factory = node_factory or AgentRuntimeFactory(tool_catalog_store=tools, model_connection_store=model_connections)
    workflow_runtime_factory = WorkflowNodeRuntimeFactory(tools, model_connections)
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

    @app.post("/mcp/demo")
    async def local_demo_mcp(request: Request) -> Dict[str, Any]:
        """无外部权限的本地 MCP JSON-RPC 演示端点，用于验证接入链路。"""
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        method = str(payload.get("method") or "tools/list")
        request_id = payload.get("id")
        if method == "tools/list":
            result: Dict[str, Any] = {
                "tools": [
                    {"name": "preview_email", "description": "仅生成邮件预览，不发送真实邮件", "inputSchema": {"type": "object"}},
                    {"name": "lookup_demo", "description": "返回本地演示检索结果", "inputSchema": {"type": "object"}},
                ]
            }
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": f"本地 MCP 已收到：{json.dumps(payload.get('params') or {}, ensure_ascii=False)}"}]}
        else:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "method not found"}}
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

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
        _migrate_ownership(user.id)
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

    def _request_user_id(request: Request) -> str:
        user = getattr(request.state, "user", None)
        return user.id if user is not None else "local-user"

    def _migrate_ownership(owner_id: str) -> None:
        for record in applications.list():
            if not record.owner_user_id:
                record.owner_user_id=owner_id;applications.save(record)
        for record in skills.list():
            if record.visibility!="builtin" and not record.owner_user_id:
                record.owner_user_id=owner_id;record.visibility="private";skills.save(record,versioned=False)
        migration_marker.write_text(owner_id,encoding="utf-8")

    def _owned_application(app_id: str, request: Request) -> ApplicationRecord:
        record = applications.get(app_id)
        if record.owner_user_id and record.owner_user_id != _request_user_id(request):
            raise HTTPException(status_code=404, detail="application not found")
        if not record.owner_user_id:
            record.owner_user_id=_request_user_id(request);applications.save(record)
        return record

    def _visible_skill(skill_id: str, request: Request, *, mutable: bool = False):
        record = skills.get(skill_id)
        if record.visibility == "builtin":
            if mutable:
                raise HTTPException(status_code=403, detail="平台内置 Skill 为只读资源")
            return record
        if record.owner_user_id != _request_user_id(request):
            raise HTTPException(status_code=404, detail="skill not found")
        return record

    def _validate_skill_selection(skill_ids: List[str], request: Request) -> None:
        for skill_id in skill_ids:
            try:
                record = _visible_skill(skill_id, request)
            except (KeyError, HTTPException) as exc:
                raise ValueError(f"Skill {skill_id} 不存在或当前用户无权使用") from exc
            if record.status != SkillStatus.PUBLISHED:
                raise ValueError(f"Skill {record.name} 尚未发布，不能绑定到应用")

    def _owned_app_for_workflow(workflow_id: str, request: Request) -> Optional[ApplicationRecord]:
        record = next((item for item in applications.list() if item.workflow_id == workflow_id), None)
        if record is not None:
            return _owned_application(record.id, request)
        return None

    def _owned_run(run_id: str, request: Request):
        record = runs.get(run_id)
        owner_id = str(record.metadata.get("owner_user_id") or "")
        if owner_id and owner_id != _request_user_id(request):
            raise HTTPException(status_code=404, detail="run not found")
        return record

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

    def _validate_application_workflow(graph: Dict[str, Any], *, runnable: bool = False) -> List[str]:
        """Validate the product workflow contract without breaking legacy graphs."""
        agents = [item for item in graph.get("agents", []) if isinstance(item, dict)]
        connections = [item for item in graph.get("connections", []) if isinstance(item, dict)]
        errors: List[str] = []
        ids = {str(item.get("id")) for item in agents}
        names = [str(item.get("name") or "").strip() for item in agents]
        kinds = [str((item.get("config") or {}).get("node_kind") or "agent") for item in agents]
        starts = [item for item, kind in zip(agents, kinds) if kind == "start"]
        ends = [item for item, kind in zip(agents, kinds) if kind == "end"]
        if len(starts) != 1:
            errors.append("工作流必须且只能包含一个开始节点")
        if len(ends) != 1:
            errors.append("工作流必须且只能包含一个结束节点")
        if any(not name for name in names) or len(names) != len(set(names)):
            errors.append("节点名称不能为空且必须唯一")
        if starts and graph.get("entry") != starts[0].get("id"):
            errors.append("工作流入口必须指向开始节点")
        for edge in connections:
            source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
            if source not in ids or (target != "END" and target not in ids):
                errors.append("工作流包含指向不存在节点的连线")
            if source == target:
                errors.append("节点不能连接到自身")
        edge_keys = [(str(edge.get("source")), str(edge.get("target")), bool(edge.get("conditional"))) for edge in connections]
        if len(edge_keys) != len(set(edge_keys)):
            errors.append("工作流不能包含重复连线")
        if starts and any(edge.get("target") == starts[0].get("id") for edge in connections):
            errors.append("开始节点不能有入边")
        if ends and any(edge.get("source") == ends[0].get("id") for edge in connections):
            errors.append("结束节点不能有出边")
        if starts:
            adjacency: Dict[str, set[str]] = {node_id: set() for node_id in ids}
            for edge in connections:
                source = str(edge.get("source") or "")
                targets = list((edge.get("path_map") or {}).values()) if edge.get("conditional") else [edge.get("target")]
                adjacency.setdefault(source, set()).update(str(target) for target in targets if target in ids)
            reached, pending = set(), [str(starts[0].get("id"))]
            while pending:
                current_id = pending.pop()
                if current_id in reached:
                    continue
                reached.add(current_id)
                pending.extend(adjacency.get(current_id, set()) - reached)
            if reached != ids:
                errors.append("所有节点必须能够从开始节点到达")
            if ends and str(ends[0].get("id")) not in reached:
                errors.append("结束节点必须能够从开始节点到达")
        if runnable and "knowledge" in kinds:
            errors.append("知识库节点已完成配置保存，但检索运行能力尚未接入")
        for item, kind in zip(agents, kinds):
            config = item.get("config") or {}
            if kind == "tool" and not config.get("tool_id"):
                errors.append(f"工具节点“{item.get('name')}”尚未选择工具")
            if kind in {"condition", "intent", "loop"} and not any(edge.get("source") == item.get("id") and edge.get("conditional") for edge in connections):
                errors.append(f"逻辑节点“{item.get('name')}”尚未配置分支连线")
        return list(dict.fromkeys(errors))

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

    def _validated_variables(definitions: List[Dict[str, Any]], supplied: Any) -> Dict[str, Any]:
        values = supplied if isinstance(supplied, dict) else {}
        result: Dict[str, Any] = {}
        seen = set()
        for definition in definitions:
            name = str(definition.get("name") or "").strip()
            if not name or not name.replace("_", "a").isalnum() or name[0].isdigit() or name in seen:
                raise ValueError("提示词变量名称必须唯一且只能包含字母、数字和下划线")
            seen.add(name)
            value = values.get(name, definition.get("default"))
            if definition.get("required") and value in {None, ""}:
                raise ValueError(f"缺少必填提示词变量：{name}")
            kind = str(definition.get("type") or "string")
            if value not in {None, ""}:
                try:
                    if kind == "number": value = float(value)
                    elif kind == "boolean": value = value if isinstance(value, bool) else str(value).lower() in {"1","true","yes","on"}
                    else: value = str(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"提示词变量 {name} 的类型不正确") from exc
            result[name] = value
        return result

    def _validate_variable_definitions(definitions: List[Dict[str, Any]]) -> None:
        seen = set()
        for definition in definitions:
            name = str(definition.get("name") or "").strip()
            if not name or not name.replace("_", "a").isalnum() or name[0].isdigit() or name in seen:
                raise ValueError("提示词变量名称必须唯一且只能包含字母、数字和下划线")
            if str(definition.get("type") or "string") not in {"string", "text", "number", "boolean"}:
                raise ValueError(f"提示词变量 {name} 的类型不受支持")
            seen.add(name)

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
            is_visual_workflow = any(
                (item.get("config") or {}).get("node_kind")
                for item in target.to_dict().get("agents", [])
            )
            compiled = target.build_graph(
                node_factory=workflow_runtime_factory if is_visual_workflow else runtime_factory,
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
            used_skills=[]
            for trace in skill_traces.list(record.id):
                for item in trace.payload.get("skills") or []:
                    if item.get("skill_id") and item not in used_skills:
                        used_skills.append(item)
            record.metadata = {
                **record.metadata,
                "active_agent": None,
                "duration_ms": round((time.perf_counter() - started_clock) * 1000, 2),
                "summary": _build_run_summary(record),
                "skills_used": used_skills,
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
        request: Request,
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
            record.metadata["owner_user_id"] = _request_user_id(request)
            runs.save(record)
            background_tasks.add_task(_execute_run, record)
            return record.to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/runs")
    def list_runs(request: Request, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        user_id = _request_user_id(request)
        return [item.to_dict() for item in runs.list(workflow_id=workflow_id) if str(item.metadata.get("owner_user_id") or "") in {"", user_id}]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, request: Request) -> Dict[str, Any]:
        try:
            return _owned_run(run_id, request).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/runs/{run_id}/events")
    async def stream_run_events(run_id: str, request: Request, after: int = 0):
        """用 SSE 推送持久化运行事件；断线后可通过 after 继续。"""
        try:
            _owned_run(run_id, request)
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
    def cancel_run(run_id: str, req: CancelRunReq, request: Request) -> Dict[str, Any]:
        try:
            _owned_run(run_id, request)
            return runs.mark_cancel_requested(run_id, reason=req.reason).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/api/runs/{run_id}/approvals/{sequence}/approve")
    def approve_tool_call(run_id: str, sequence: int, req: ApprovalDecisionReq, request: Request) -> Dict[str, Any]:
        _owned_run(run_id, request)
        return _record_approval_decision(run_id, sequence, approved=True, reason=req.reason)

    @app.post("/api/runs/{run_id}/approvals/{sequence}/reject")
    def reject_tool_call(run_id: str, sequence: int, req: ApprovalDecisionReq, request: Request) -> Dict[str, Any]:
        _owned_run(run_id, request)
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
        tool_call = dict(target.get("tool_call") or {})
        # 高风险工具不会在模型选择阶段执行；只有批准后才在这里以绕过二次审批的方式执行。
        if approved:
            try:
                tool = tools.get(str(tool_call.get("id") or ""))
                task_text = str((tool_call.get("arguments") or {}).get("task") or record.input.get("input") or "")
                result = ToolRuntime(tools).execute(tool, task_text, bypass_approval=True).to_dict()
            except Exception as exc:  # noqa: BLE001 - 审批后的执行错误必须留在审计轨迹中
                result = {**tool_call, "status": "failed", "error": str(exc)}
            _append_event(
                record,
                {
                    "type": "tool_result",
                    "node": target.get("node"),
                    "tool_call": result,
                    "message": f"审批后已执行工具 {result.get('display_name') or result.get('name')}",
                },
            )
        else:
            _append_event(
                record,
                {
                    "type": "tool_result",
                    "node": target.get("node"),
                    "tool_call": {**tool_call, "status": "rejected", "error": reason or "用户拒绝执行"},
                    "message": "工具调用已被用户拒绝",
                },
            )
        runs.save(record)
        return record.to_dict()

    @app.post("/api/runs/{run_id}/retry")
    async def retry_run(run_id: str, background_tasks: BackgroundTasks, request: Request) -> Dict[str, Any]:
        try:
            _owned_run(run_id, request)
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
    @app.get("/api/model-presets")
    def list_model_presets() -> List[Dict[str, Any]]:
        return MODEL_PRESETS

    @app.get("/api/model-connections")
    def list_model_connections() -> List[Dict[str, Any]]:
        return [item.to_dict() for item in model_connections.list()]

    @app.post("/api/model-connections")
    def create_model_connection(req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return model_connections.create(req).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/model-connections/{connection_id}")
    def update_model_connection(connection_id: str, req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            current = model_connections.get(connection_id)
            return model_connections.save(ModelConnection.from_dict({**current.to_dict(), **req, "id": connection_id})).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/model-connections/{connection_id}/test")
    def test_model_connection(connection_id: str) -> Dict[str, Any]:
        try:
            item = model_connections.get(connection_id)
            if not item.base_url:
                raise ValueError("模型连接缺少 base_url")
            headers = {"Accept": "application/json"}
            if item.api_key_env and os.environ.get(item.api_key_env):
                headers["Authorization"] = f"Bearer {os.environ[item.api_key_env]}"
            request = urllib.request.Request(f"{item.base_url}/models", headers=headers)
            with urllib.request.urlopen(request, timeout=8) as response:  # noqa: S310 - URL is administrator configured
                if response.status >= 400: raise ValueError(f"模型服务返回 HTTP {response.status}")
            item.test_status = "succeeded"
            return model_connections.save(item).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            try:
                item.test_status = "failed"; model_connections.save(item)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail=f"模型连接测试失败：{exc}")

    @app.get("/api/apps")
    def list_applications(request: Request) -> List[Dict[str, Any]]:
        user_id=_request_user_id(request)
        return [item.to_dict() for item in applications.list() if item.owner_user_id in {"",user_id}]

    @app.post("/api/apps")
    def create_application(req: CreateApplicationReq, request: Request) -> Dict[str, Any]:
        try:
            _validate_skill_selection(req.skill_ids, request)
            _validate_variable_definitions(req.prompt_variables)
            if req.app_type not in {"agent", "workflow"}:
                raise ValueError("app_type must be 'agent' or 'workflow'")
            if req.model not in {"", "auto", "device", "edge", "cloud"}:
                selected_model = model_connections.get(req.model)
                if not selected_model.enabled or selected_model.test_status != "succeeded" or not selected_model.to_dict()["configured"]:
                    raise ValueError("指定模型连接未启用或尚未测试成功")
            app_record = applications.create(
                name=req.name,
                app_type=req.app_type,
                description=req.description,
                model=req.model,
                system_prompt=req.system_prompt,
                avatar_url=req.avatar_url,
                tool_ids=req.tool_ids,
                skill_ids=req.skill_ids,
                knowledge_base_ids=req.knowledge_base_ids,
                memory_bank_ids=req.memory_bank_ids,
                prompt_variables=req.prompt_variables,
                owner_user_id=_request_user_id(request),
                metadata=req.metadata,
            )
            draft = Orchestrator()
            if req.app_type == "workflow":
                entry_id = draft.create_agent(
                    name="开始", description="接收工作流输入", config={"node_kind": "start", "input_fields": ["input"]}
                )
                end_id = draft.create_agent(
                    name="结束", description="返回工作流最终输出", config={"node_kind": "end", "output_field": "input"}
                )
                draft.connect(entry_id, end_id)
            else:
                entry_id = draft.create_agent(
                    name=req.name,
                    sys_prompt=req.system_prompt,
                    model=req.model,
                    description=req.description or "负责应用入口任务理解、工具调用和最终答复。",
                    config={
                        "tool_ids": req.tool_ids,
                        "skill_ids": req.skill_ids,
                        "knowledge_base_ids": req.knowledge_base_ids,
                        "memory_bank_ids": req.memory_bank_ids,
                        "prompt_variables": req.prompt_variables,
                    },
                )
            draft.set_entry(entry_id)
            workflow = workflows.create(
                name=req.name,
                description=req.description,
                tags=[req.app_type, "application"],
                metadata={
                    "application_id": app_record.id,
                    **({"editor": {"positions": {entry_id: {"x": 120, "y": 240}, end_id: {"x": 520, "y": 240}}, "viewport": {"x": 0, "y": 0, "zoom": 1}}} if req.app_type == "workflow" else {}),
                    **req.metadata,
                },
                graph=draft.to_dict(),
            )
            app_record.workflow_id = workflow.id
            app_record.entry_agent_id = entry_id
            app_record.metadata = {**app_record.metadata, "workflow_version": workflow.version}
            applications.save(app_record)
            return app_record.to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/apps/{app_id}")
    def get_application(app_id: str, request: Request) -> Dict[str, Any]:
        try:
            app_record = _owned_application(app_id, request)
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
    def update_application(app_id: str, req: UpdateApplicationReq, request: Request) -> Dict[str, Any]:
        try:
            current = _owned_application(app_id, request)
            if req.skill_ids is not None:
                _validate_skill_selection(req.skill_ids, request)
            if req.prompt_variables is not None:
                _validate_variable_definitions(req.prompt_variables)
            if req.model is not None and req.model not in {"", "auto", "device", "edge", "cloud"}:
                selected_model = model_connections.get(req.model)
                if not selected_model.enabled or selected_model.test_status != "succeeded" or not selected_model.to_dict()["configured"]:
                    raise ValueError("指定模型连接未启用或尚未测试成功")
            updated = ApplicationRecord.from_dict(
                {
                    **current.to_dict(),
                    "name": req.name if req.name is not None else current.name,
                    "description": req.description if req.description is not None else current.description,
                    "status": req.status if req.status is not None else current.status,
                    "model": req.model if req.model is not None else current.model,
                    "system_prompt": req.system_prompt if req.system_prompt is not None else current.system_prompt,
                    "avatar_url": req.avatar_url if req.avatar_url is not None else current.avatar_url,
                    "tool_ids": req.tool_ids if req.tool_ids is not None else current.tool_ids,
                    "skill_ids": req.skill_ids if req.skill_ids is not None else current.skill_ids,
                    "knowledge_base_ids": req.knowledge_base_ids if req.knowledge_base_ids is not None else current.knowledge_base_ids,
                    "memory_bank_ids": req.memory_bank_ids if req.memory_bank_ids is not None else current.memory_bank_ids,
                    "prompt_variables": req.prompt_variables if req.prompt_variables is not None else current.prompt_variables,
                    "metadata": req.metadata if req.metadata is not None else current.metadata,
                }
            )
            if updated.workflow_id:
                workflow = workflows.get(updated.workflow_id)
                graph = Orchestrator.from_dict(workflow.graph)
                if updated.entry_agent_id and updated.app_type == "agent":
                    graph.update_agent(
                        updated.entry_agent_id,
                        name=updated.name,
                        sys_prompt=updated.system_prompt,
                        model=updated.model,
                        description=updated.description,
                        config={
                            "tool_ids": updated.tool_ids,
                            "skill_ids": updated.skill_ids,
                            "knowledge_base_ids": updated.knowledge_base_ids,
                            "memory_bank_ids": updated.memory_bank_ids,
                            "prompt_variables": updated.prompt_variables,
                        },
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
    def publish_application(app_id: str, request: Request) -> Dict[str, Any]:
        try:
            app_record = _owned_application(app_id, request)
            if app_record.app_type == "workflow":
                workflow = workflows.get(app_record.workflow_id)
                errors = _validate_application_workflow(workflow.graph, runnable=True)
                if errors:
                    raise ValueError("；".join(errors))
            app_record.status = "published"
            return applications.save(app_record).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/apps/{app_id}/runs")
    async def create_application_run(
        app_id: str,
        req: CreateApplicationRunReq,
        background_tasks: BackgroundTasks,
        request: Request,
    ) -> Dict[str, Any]:
        """以应用为入口发起调试运行，前端无需理解内部 workflow_id。"""
        try:
            app_record = _owned_application(app_id, request)
            if not app_record.workflow_id:
                raise HTTPException(status_code=400, detail="应用尚未绑定工作流")
            workflows.get(app_record.workflow_id)
            if app_record.app_type == "workflow":
                errors = _validate_application_workflow(workflows.get(app_record.workflow_id).graph, runnable=True)
                if errors:
                    raise HTTPException(status_code=400, detail="；".join(errors))
            input_payload = dict(req.input or {})
            try:
                input_payload["variables"] = _validated_variables(app_record.prompt_variables, input_payload.get("variables"))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            if "input" not in input_payload:
                input_payload["input"] = f"请运行应用：{app_record.name}"
            record = runs.create(
                input=input_payload,
                recursion_limit=req.recursion_limit,
                workflow_id=app_record.workflow_id,
            )
            record.metadata["application_id"] = app_record.id
            record.metadata["application_name"] = app_record.name
            record.metadata["owner_user_id"] = app_record.owner_user_id
            record.status = "queued"
            runs.save(record)
            background_tasks.add_task(_execute_run, record)
            return record.to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/apps/{app_id}/runs")
    def list_application_runs(app_id: str, request: Request) -> List[Dict[str, Any]]:
        """列出某个应用触发的运行记录，便于应用详情页做调试历史。"""
        try:
            app_record = _owned_application(app_id, request)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return [
            item.to_dict()
            for item in runs.list(workflow_id=app_record.workflow_id)
            if item.metadata.get("application_id") in {None, app_record.id}
        ]

    @app.delete("/api/apps/{app_id}")
    def delete_application(app_id: str, request: Request) -> Dict[str, Any]:
        try:
            _owned_application(app_id, request)
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
    def list_workflows(request: Request) -> List[Dict[str, Any]]:
        user_id = _request_user_id(request)
        owned_workflow_ids = {item.workflow_id for item in applications.list() if item.owner_user_id in {"", user_id}}
        foreign_workflow_ids = {item.workflow_id for item in applications.list() if item.owner_user_id not in {"", user_id}}
        return [item.to_dict() for item in workflows.list() if item.id in owned_workflow_ids or item.id not in foreign_workflow_ids]

    @app.get("/api/workflows/{workflow_id}")
    def get_workflow(workflow_id: str, request: Request) -> Dict[str, Any]:
        try:
            _owned_app_for_workflow(workflow_id, request)
            return workflows.get(workflow_id).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/workflows/{workflow_id}/versions")
    def list_workflow_versions(workflow_id: str, request: Request) -> List[Dict[str, Any]]:
        try:
            _owned_app_for_workflow(workflow_id, request)
            return [item.to_dict() for item in workflows.list_versions(workflow_id)]
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/workflows/{workflow_id}/versions/{version}")
    def get_workflow_version(workflow_id: str, version: int, request: Request) -> Dict[str, Any]:
        try:
            _owned_app_for_workflow(workflow_id, request)
            return workflows.get_version(workflow_id, version).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.put("/api/workflows/{workflow_id}")
    def update_workflow(workflow_id: str, req: UpdateWorkflowReq, request: Request) -> Dict[str, Any]:
        try:
            _owned_app_for_workflow(workflow_id, request)
            current = workflows.get(workflow_id)
            next_graph = req.graph if req.graph is not None else current.graph
            if "workflow" in current.tags:
                errors = _validate_application_workflow(next_graph)
                if errors:
                    raise ValueError("；".join(errors))
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
                    "graph": next_graph,
                }
            )
            return workflows.save(updated).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/workflows/{workflow_id}/load")
    def load_workflow(workflow_id: str, request: Request) -> Dict[str, Any]:
        nonlocal orch
        try:
            _owned_app_for_workflow(workflow_id, request)
            record = workflows.get(workflow_id)
            orch = Orchestrator.from_dict(record.graph)
            _apply_policy(orch)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"ok": True, "workflow": record.to_dict(), "graph": orch.to_dict()}

    @app.post("/api/workflows/{workflow_id}/rollback/{version}")
    def rollback_workflow(workflow_id: str, version: int, request: Request) -> Dict[str, Any]:
        try:
            _owned_app_for_workflow(workflow_id, request)
            return workflows.rollback(workflow_id, version).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/workflows/{workflow_id}")
    def delete_workflow(workflow_id: str, request: Request) -> Dict[str, Any]:
        try:
            _owned_app_for_workflow(workflow_id, request)
            workflows.delete(workflow_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"ok": True}

    # ------------------------- 技能生命周期 ------------------------- #
    @app.post("/api/skills")
    def create_skill(req: CreateSkillReq, request: Request) -> Dict[str, Any]:
        try:
            return skills.create(
                name=req.name,
                content=req.content,
                description=req.description,
                tags=req.tags,
                metadata={**req.metadata,"source":"manual","validation":{"passed":True,"mode":"automatic"}},
                status=SkillStatus.PUBLISHED,
                owner_user_id=_request_user_id(request),
                visibility="private",
                source_type="manual",
                validation_status="passed",
            ).to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/import/git")
    def import_skill_git(req: Dict[str, Any], request: Request) -> Dict[str, Any]:
        try:
            url = str(req.get("url") or "")
            package = read_skill_git(url)
            record = skills.create(name=package["name"],content=package["content"],description=str(req.get("description") or "从 Git 仓库导入"),status=SkillStatus.PUBLISHED,tags=["imported","git"],metadata={"source":"git","source_url":url,"revision":package["sha256"],"references":package["references"],"scripts_ignored":True,"validation":{"passed":True,"mode":"automatic"}},owner_user_id=_request_user_id(request),visibility="private",source_type="git",validation_status="passed",package_sha256=package["sha256"])
            return record.to_dict()
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/skills/import/zip")
    async def import_skill_zip(request: Request) -> Dict[str, Any]:
        try:
            package = read_skill_zip(await request.body())
            record = skills.create(name=package["name"],content=package["content"],description=request.headers.get("x-skill-description","从 ZIP 包导入"),status=SkillStatus.PUBLISHED,tags=["imported","zip"],metadata={"source":"zip","sha256":package["sha256"],"references":package["references"],"scripts_ignored":True,"validation":{"passed":True,"mode":"automatic"}},owner_user_id=_request_user_id(request),visibility="private",source_type="zip",validation_status="passed",package_sha256=package["sha256"])
            return record.to_dict()
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/skills")
    def list_skills(request: Request, status: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            user_id=_request_user_id(request)
            return [item.to_dict() for item in skills.list(status=status) if item.visibility=="builtin" or item.owner_user_id==user_id]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/search")
    def search_skills(req: SkillSearchReq, request: Request) -> Dict[str, Any]:
        matches = skill_retriever.retrieve(
            req.query,
            node=req.node,
            metadata=req.metadata,
            top_k=req.top_k,
        )
        visible={item["id"] for item in list_skills(request)}
        return {"matches": [item.to_dict() for item in matches if item.skill.id in visible]}

    @app.put("/api/skills/{skill_id}")
    def update_private_skill(skill_id: str, req: CreateSkillReq, request: Request) -> Dict[str, Any]:
        try:
            record=_visible_skill(skill_id,request,mutable=True)
            record.name=req.name;record.content=req.content;record.description=req.description;record.tags=req.tags;record.metadata={**record.metadata,**req.metadata,"validation":{"passed":True,"mode":"automatic"}};record.status=SkillStatus.PUBLISHED;record.validation_status="passed"
            return skills.save(record).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404,detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400,detail=str(exc))

    @app.delete("/api/skills/{skill_id}")
    def delete_private_skill(skill_id: str, request: Request) -> Dict[str, Any]:
        try:
            _visible_skill(skill_id,request,mutable=True);skills.delete(skill_id);return {"ok":True}
        except KeyError as exc:
            raise HTTPException(status_code=404,detail=str(exc))

    @app.get("/api/skills/{skill_id}")
    def get_skill(skill_id: str, request: Request) -> Dict[str, Any]:
        try:
            return _visible_skill(skill_id, request).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/skills/{skill_id}/versions")
    def list_skill_versions(skill_id: str, request: Request) -> List[Dict[str, Any]]:
        try:
            _visible_skill(skill_id, request)
            return [item.to_dict() for item in skills.list_versions(skill_id)]
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/skills/{skill_id}/versions/{version}")
    def get_skill_version(skill_id: str, version: int, request: Request) -> Dict[str, Any]:
        try:
            _visible_skill(skill_id, request)
            return skills.get_version(skill_id, version).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/candidates/from-run")
    def create_skill_candidate(req: CreateSkillCandidateReq, request: Request) -> Dict[str, Any]:
        try:
            record = skill_evolution.create_candidate_from_run(
                req.run_id,
                name=req.name,
                description=req.description,
                tags=req.tags,
                metadata=req.metadata,
            )
            record.owner_user_id=_request_user_id(request);record.visibility="private";record.source_type="run";skills.save(record)
            report=skill_evolution.validate(record.id)
            if not report.passed:
                raise ValueError("Skill 自动验证失败："+"；".join(report.findings))
            published=skill_evolution.publish(record.id,approved_by=_request_user_id(request))
            published.validation_status="passed";published.metadata={**published.metadata,"validation_mode":"automatic"}
            return skills.save(published).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/{skill_id}/validate")
    def validate_skill(skill_id: str, request: Request) -> Dict[str, Any]:
        try:
            _visible_skill(skill_id, request, mutable=True)
            return skill_evolution.validate(skill_id).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/{skill_id}/publish")
    def publish_skill(skill_id: str, req: SkillDecisionReq, request: Request) -> Dict[str, Any]:
        try:
            _visible_skill(skill_id, request, mutable=True)
            return skill_evolution.publish(
                skill_id,
                approved_by=req.approved_by or "",
            ).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/{skill_id}/reject")
    def reject_skill(skill_id: str, req: SkillDecisionReq, request: Request) -> Dict[str, Any]:
        try:
            _visible_skill(skill_id, request, mutable=True)
            return skill_evolution.reject(skill_id, reason=req.reason).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/{skill_id}/retire")
    def retire_skill(skill_id: str, req: SkillDecisionReq, request: Request) -> Dict[str, Any]:
        try:
            _visible_skill(skill_id, request, mutable=True)
            return skill_evolution.retire(skill_id, reason=req.reason).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/skills/{skill_id}/rollback/{version}")
    def rollback_skill(skill_id: str, version: int, req: SkillDecisionReq, request: Request) -> Dict[str, Any]:
        try:
            _visible_skill(skill_id, request, mutable=True)
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
    def set_skill_rollout(skill_id: str, req: SkillRolloutReq, request: Request) -> Dict[str, Any]:
        try:
            _visible_skill(skill_id, request, mutable=True)
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
    def get_skill_traces(run_id: str, request: Request) -> Dict[str, Any]:
        _owned_run(run_id, request)
        return {"events": [item.to_dict() for item in skill_traces.list(run_id)]}

    # ------------------------- 百炼式资源市场 ------------------------- #
    # 这些条目是可安装的本地模板，不声明为已接入第三方云服务。
    mcp_market = [
        {"slug": "local-demo", "name": "本地演示 MCP", "provider": "AgentForge", "category": "演示", "description": "零权限 JSON-RPC 演示服务，用于验证 MCP URL、工具发现与调用链。", "installs": 1, "cover": "violet", "mcp_url": "http://127.0.0.1:8000/mcp/demo"},
        {"slug": "web-search", "name": "联网检索", "provider": "示例市场", "category": "通用办公", "description": "为 Agent 提供检索与网页摘要能力；安装后仍需配置实际 MCP URL。", "installs": 129, "cover": "mint"},
        {"slug": "calendar", "name": "日历与会议", "provider": "示例市场", "category": "通用办公", "description": "查询可用时间、创建会议和发送日程提醒的 MCP 接入模板。", "installs": 88, "cover": "violet"},
        {"slug": "email", "name": "邮件发送", "provider": "示例市场", "category": "通用办公", "description": "起草、确认并发送邮件。真实发送前将进入工具审批流程。", "installs": 201, "cover": "blue"},
        {"slug": "maps", "name": "地图与路线", "provider": "示例市场", "category": "生活服务", "description": "地点搜索、路线规划和行程建议的 MCP 接入模板。", "installs": 109, "cover": "warm"},
        {"slug": "contract", "name": "合同信息抽取", "provider": "示例市场", "category": "法律", "description": "从合同正文中抽取关键字段；可替换为企业内的 MCP 服务地址。", "installs": 66, "cover": "rose"},
        {"slug": "knowledge", "name": "企业知识检索", "provider": "示例市场", "category": "知识库", "description": "面向企业文档的检索问答 MCP 接入模板。", "installs": 55, "cover": "cyan"},
    ]
    skill_market = [
        {"slug": "email-writer", "name": "商务邮件撰写", "category": "通用办公", "description": "根据收件人、目的和语气起草清晰、可发送的商务邮件。", "content": "# 商务邮件撰写\n\n先确认收件人、主题、目的和语气；给出结构化邮件草稿。发送前必须请求用户确认。"},
        {"slug": "research-report", "name": "研究报告", "category": "内容创意", "description": "把研究主题拆解为目标、证据、结论与待验证项，避免虚构来源。", "content": "# 研究报告\n\n先列出研究问题和证据需求，输出结论时标识事实、推断和待核验项。"},
        {"slug": "travel-planner", "name": "旅行计划", "category": "通用办公", "description": "生成兼顾时间、预算、天气与交通的行程方案。", "content": "# 旅行计划\n\n确认目的地、日期、预算、同行人和偏好；涉及实时信息时建议调用已授权工具。"},
        {"slug": "meeting-summary", "name": "会议纪要", "category": "通用办公", "description": "将会议材料整理为结论、行动项、负责人和截止时间。", "content": "# 会议纪要\n\n以结论、行动项、负责人、截止时间四部分输出；缺失信息明确标记待补充。"},
        {"slug": "web-design", "name": "网页设计", "category": "代码开发", "description": "把用户需求转为信息架构、界面层级和可实施的前端建议。", "content": "# 网页设计\n\n先给出页面目标、用户路径和组件清单，再输出可实施的视觉与交互建议。"},
        {"slug": "data-analysis", "name": "数据分析", "category": "金融", "description": "帮助解释指标、识别异常并给出可复现的分析路径。", "content": "# 数据分析\n\n明确数据范围与口径，区分计算结果和业务推断，给出复核步骤。"},
    ]
    skill_market = BUILTIN_SKILLS
    app_templates = [
        {"slug": "blank-agent", "name": "空白智能体", "description": "最小化核心工具集，从零开始构建。", "system_prompt": "你是可靠的智能体助手。先澄清任务，再规划、执行和总结。"},
        {"slug": "article-polish", "name": "文章润色", "description": "改善表达和结构，不改变原意、不捏造事实。", "system_prompt": "你是文章润色助手。保留事实和原意，输出修改稿与修改说明。"},
        {"slug": "research", "name": "研究报告", "description": "从主题到完整报告的研究型智能体模板。", "system_prompt": "你是研究报告助手。先分解问题与证据，再输出有来源边界的完整报告。"},
        {"slug": "email-assistant", "name": "邮件助手", "description": "起草邮件并在真实发送前请求用户确认。", "system_prompt": "你是邮件助手。先收集收件人、主题、正文和附件信息；调用发送邮件工具前必须等待用户批准。"},
    ]

    @app.get("/api/marketplace/mcp")
    def list_mcp_marketplace() -> List[Dict[str, Any]]:
        return mcp_market

    @app.post("/api/marketplace/mcp/{slug}/install")
    def install_mcp_template(slug: str) -> Dict[str, Any]:
        item = next((x for x in mcp_market if x["slug"] == slug), None)
        if item is None:
            raise HTTPException(status_code=404, detail="未找到 MCP 市场模板")
        existing = next((x for x in tools.list() if x.name == f"mcp_{slug}"), None)
        if existing is not None:
            return {"installed": False, "tool": existing.to_dict(), "message": "该 MCP 模板已经安装"}
        record = tools.create(
            name=f"mcp_{slug}", display_name=item["name"], description=item["description"],
            category="mcp", tags=["mcp", item["category"]],
            metadata={"source": "mcp", "adapter": "mcp_http", "market_slug": slug, "risk": "read", "mcp_url": item.get("mcp_url", ""), "needs_configuration": not bool(item.get("mcp_url"))},
        )
        message = "本地演示 MCP 已安装，可直接在 MCP 管理中测试" if item.get("mcp_url") else "MCP 模板已安装，请在 MCP 管理中填写服务地址"
        return {"installed": True, "tool": record.to_dict(), "message": message}

    @app.get("/api/marketplace/skills")
    def list_skill_marketplace() -> List[Dict[str, Any]]:
        records={item.metadata.get("market_slug"):item for item in skills.list() if item.visibility=="builtin"}
        return [{**{k:v for k,v in item.items() if k!="content"},"skill_id":records[item["slug"]].id if item["slug"] in records else "","status":"published","version":records[item["slug"]].version if item["slug"] in records else 1,"source_type":"builtin"} for item in skill_market]

    @app.post("/api/marketplace/skills/{slug}/install")
    def install_skill_template(slug: str) -> Dict[str, Any]:
        item = next((x for x in skill_market if x["slug"] == slug), None)
        if item is None:
            raise HTTPException(status_code=404, detail="未找到 Skill 市场模板")
        existing = next((x for x in skills.list() if x.metadata.get("market_slug") == slug), None)
        if existing is not None:
            return {"installed": False, "skill": existing.to_dict(), "message": "平台内置 Skill 已可直接使用"}
        record = skills.create(name=item["name"],content=item["content"],description=item["description"],tags=[item["category"],"market","builtin"],status=SkillStatus.PUBLISHED,visibility="builtin",source_type="builtin",validation_status="passed",metadata={"market_slug":slug,"source":"builtin","category":item["category"]})
        return {"installed": True, "skill": record.to_dict(), "message": "平台内置 Skill 已可直接使用"}

    @app.get("/api/marketplace/apps")
    def list_application_templates() -> List[Dict[str, Any]]:
        return app_templates

    @app.post("/api/marketplace/apps/{slug}/install")
    def install_application_template(slug: str) -> Dict[str, Any]:
        item = next((x for x in app_templates if x["slug"] == slug), None)
        if item is None:
            raise HTTPException(status_code=404, detail="未找到应用模板")
        app_record = applications.create(name=item["name"], app_type="agent", description=item["description"], system_prompt=item["system_prompt"], metadata={"template_slug": slug})
        draft = Orchestrator()
        entry_id = draft.create_agent(name=item["name"], sys_prompt=item["system_prompt"], description=item["description"], config={"tool_ids": [], "skill_ids": []})
        draft.set_entry(entry_id)
        workflow = workflows.create(name=item["name"], description=item["description"], tags=["agent", "template"], metadata={"application_id": app_record.id, "template_slug": slug}, graph=draft.to_dict())
        app_record.workflow_id, app_record.entry_agent_id = workflow.id, entry_id
        return {"installed": True, "application": applications.save(app_record).to_dict()}

    # ------------------------- 记忆库目录 ------------------------- #
    @app.get("/api/memory-banks")
    def list_memory_banks() -> List[Dict[str, Any]]:
        return [item.to_dict() for item in memory_banks.list()]

    @app.post("/api/memory-banks")
    def create_memory_bank(req: CreateMemoryBankReq) -> Dict[str, Any]:
        try:
            return memory_banks.create(name=req.name, description=req.description, metadata=req.metadata).to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/memory-banks/{bank_id}")
    def delete_memory_bank(bank_id: str) -> Dict[str, Any]:
        try:
            memory_banks.delete(bank_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"ok": True}

    # ------------------------- 组件与数据资源 ------------------------- #
    resource_kinds = {"components", "knowledge-bases", "data-connections", "evaluations", "observability", "permissions", "ui-designs"}
    component_market = [
        {"slug": "web-search-node", "name": "网页检索组件", "category": "工具", "description": "在工作流中提供联网检索节点；需要挂载已安装的检索 MCP。"},
        {"slug": "approval-node", "name": "人工审批组件", "category": "安全", "description": "在高风险工具调用或发布操作前暂停并等待用户决定。"},
        {"slug": "summary-node", "name": "任务总结组件", "category": "输出", "description": "根据运行轨迹生成可审计的最终总结。"},
        {"slug": "todo-node", "name": "TODO 规划组件", "category": "编排", "description": "使用现有 TODO 管理器拆解并跟踪长任务，降低上下文漂移。"},
    ]

    @app.get("/api/components/market")
    def list_component_market() -> List[Dict[str, Any]]:
        return component_market

    @app.post("/api/components/{slug}/install")
    def install_component(slug: str) -> Dict[str, Any]:
        item = next((x for x in component_market if x["slug"] == slug), None)
        if item is None:
            raise HTTPException(status_code=404, detail="未找到组件模板")
        existing = next((x for x in console_resources.list("components") if x.metadata.get("market_slug") == slug), None)
        if existing is not None:
            return {"installed": False, "component": existing.to_dict(), "message": "该组件已经安装"}
        record = console_resources.create(kind="components", name=item["name"], description=item["description"], metadata={"market_slug": slug, "category": item["category"], "source": "market"})
        return {"installed": True, "component": record.to_dict(), "message": "组件已安装，可在组件管理中查看"}

    @app.get("/api/resources/{kind}")
    def list_console_resources(kind: str) -> List[Dict[str, Any]]:
        if kind not in resource_kinds:
            raise HTTPException(status_code=404, detail="不支持的资源类型")
        return [item.to_dict() for item in console_resources.list(kind)]

    @app.post("/api/resources/{kind}")
    def create_console_resource(kind: str, req: CreateConsoleResourceReq) -> Dict[str, Any]:
        if kind not in resource_kinds:
            raise HTTPException(status_code=404, detail="不支持的资源类型")
        try:
            return console_resources.create(kind=kind, name=req.name, description=req.description, metadata=req.metadata).to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/resources/{kind}/{resource_id}")
    def delete_console_resource(kind: str, resource_id: str) -> Dict[str, Any]:
        if kind not in resource_kinds:
            raise HTTPException(status_code=404, detail="不支持的资源类型")
        try:
            console_resources.delete(kind, resource_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"ok": True}

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
    @app.post("/api/tool-connections/mcp")
    def import_mcp_connection(req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            url = validate_remote_url(str(req.get("url") or ""))
            name = str(req.get("name") or "Remote MCP").strip()
            credential_env = str(req.get("credential_env") or "")
            timeout = min(30, max(1, int(req.get("timeout_seconds") or 8)))
            connection_id = f"mcp-{int(time.time())}"
            imported = []
            for remote in discover_mcp_tools(url, credential_env, timeout):
                remote_name = str(remote["name"])
                slug = f"mcp_{connection_id}_{remote_name}".replace("-", "_")
                metadata = {"source":"mcp","adapter":"mcp_http","connection_id":connection_id,"mcp_url":url,"method":"tools/call","remote_tool_name":remote_name,"input_schema":remote.get("inputSchema") or {},"credential_env":credential_env,"risk":str(req.get("risk") or "read"),"sync_status":"synced","timeout_seconds":timeout}
                imported.append(tools.create(name=slug,display_name=str(remote.get("title") or remote_name),description=str(remote.get("description") or f"{name} MCP 工具"),category="mcp",tags=["mcp","external"],metadata=metadata).to_dict())
            return {"connection_id":connection_id,"tools":imported}
        except (ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/tool-connections/openapi")
    def import_openapi_connection(req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            source_url = str(req.get("url") or "")
            content = read_remote_document(source_url) if source_url else json.dumps(req.get("document") or {}).encode("utf-8")
            document = parse_openapi(content)
            imported = []
            for operation in openapi_operations(document, source_url or "https://configured.invalid/openapi.json", str(req.get("credential_env") or "")):
                existing = next((item for item in tools.list() if item.name == operation["name"]), None)
                if existing:
                    existing.display_name=operation["display_name"];existing.description=operation["description"];existing.metadata={**existing.metadata,**operation["metadata"]}; imported.append(tools.save(existing).to_dict())
                else:
                    imported.append(tools.create(name=operation["name"],display_name=operation["display_name"],description=operation["description"],category="openapi",tags=["openapi","external"],metadata=operation["metadata"]).to_dict())
            return {"connection_id":f"openapi-{int(time.time())}","tools":imported}
        except (ValueError, urllib.error.URLError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/tool-connections/{connection_id}/sync")
    def sync_tool_connection(connection_id: str) -> Dict[str, Any]:
        try:
            connected = [item for item in tools.list() if item.metadata.get("connection_id") == connection_id]
            if not connected:
                raise KeyError(f"tool connection not found: {connection_id}")
            seed = connected[0]
            remote_tools = discover_mcp_tools(str(seed.metadata["mcp_url"]), str(seed.metadata.get("credential_env") or ""), float(seed.metadata.get("timeout_seconds") or 8))
            synced = []
            by_name = {str(item.metadata.get("remote_tool_name")): item for item in connected}
            for remote in remote_tools:
                remote_name = str(remote["name"])
                item = by_name.get(remote_name)
                metadata = {**seed.metadata,"remote_tool_name":remote_name,"input_schema":remote.get("inputSchema") or {},"sync_status":"synced"}
                if item:
                    item.display_name=str(remote.get("title") or remote_name);item.description=str(remote.get("description") or item.description);item.metadata=metadata;synced.append(tools.save(item).to_dict())
                else:
                    slug=f"mcp_{connection_id}_{remote_name}".replace("-","_");synced.append(tools.create(name=slug,display_name=str(remote.get("title") or remote_name),description=str(remote.get("description") or "MCP 工具"),category="mcp",tags=["mcp","external"],metadata=metadata).to_dict())
            return {"connection_id":connection_id,"tools":synced,"status":"synced"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

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

    @app.post("/api/tools/{tool_id}/test")
    def test_tool(tool_id: str, req: TestToolReq) -> Dict[str, Any]:
        try:
            tool = tools.get(tool_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return ToolRuntime(tools).execute(tool, req.task).to_dict()

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
