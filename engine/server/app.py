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
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response, UploadFile, File, Form
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
from ..modules.conversations import ConversationStore, MemoryAuditStore, SENSITIVE_PATTERN, build_conversation_context, durable_memory_candidates
from ..modules.context import ContextPolicy
from ..modules.context.todo import TodoManager
from ..modules.mcp_integration import (
    MCPConfigStore,
    MCPToolRecord,
    list_mcp_tools,
    test_mcp_connection,
)
from ..modules.product_ops import (
    ApiKeyStore,
    ApplicationRecord,
    ApplicationStore,
    normalize_memory_config,
    ConsoleResourceStore,
    MemoryBankStore,
    normalize_memory_retrieval_config,
    ProjectSnapshotService,
    ProductStatusService,
    ToolCatalogStore,
    ToolConnectionRecord,
    ToolConnectionStore,
    ToolRecord,
)
from ..modules.model_connections import MODEL_PRESETS, ModelConnection, ModelConnectionStore, connection_api_key
from ..modules.external_imports import discover_mcp_tools, openapi_operations, parse_openapi, read_remote_document, read_skill_file, read_skill_git, read_skill_zip, validate_remote_url
from ..modules.memory import HybridTieredMemoryStore, MemoryBankRuntime, MemoryContext, MemoryScope, list_bank_memories
from ..modules.security_ops import ApiAuditRecord, ApiAuditStore, utc_now
from ..modules.skills import (
    SkillEvolutionService,
    SkillRepository,
    SkillRetriever,
    SkillStatus,
    SkillTraceStore,
)
from ..modules.workflows import RunRecord, RunStore, WorkflowRecord, WorkflowStore
from ..modules.tools import GMAIL_STATIC_OAUTH_SCHEMA, MCPAuthorizationRequired, MCPAuthorizationStore, ToolRuntime, complete_authorization, ensure_builtin_tools, inspect_connection, missing_required, start_authorization
from ..modules.workflow_runtime import WorkflowNodeRuntimeFactory
from ..modules.knowledge import KnowledgeStore
from ..modules.workspace_tools import WorkspaceStore
from ..modules.file_preview import preview_run_file
from ..orchestrator import NodeFactory, Orchestrator, _load_dotenv_for_context_policy


def _choose_local_directory() -> str:
    """Open the host OS directory picker; the browser cannot expose absolute paths."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境不支持系统文件夹选择器") from exc
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        return str(filedialog.askdirectory(parent=root, title="选择本地代码工作区", mustexist=True) or "")
    finally:
        root.destroy()


BUILTIN_SKILLS = [
    {"slug":"email-writer","name":"商务邮件撰写","category":"通用办公","description":"根据收件人、目的和语气起草清晰、可发送的商务邮件。","content":"# 商务邮件撰写\n\n先确认收件人、主题、目的和语气；给出结构化邮件草稿。发送前必须请求用户确认。"},
    {"slug":"research-report","name":"研究报告","category":"内容创意","description":"把研究主题拆解为目标、证据、结论与待验证项，避免虚构来源。","content":"# 研究报告\n\n先列出研究问题和证据需求，输出结论时标识事实、推断和待核验项。"},
    {"slug":"travel-planner","name":"旅行计划","category":"通用办公","description":"生成兼顾时间、预算、天气与交通的行程方案。","content":"# 旅行计划\n\n确认目的地、日期、预算、同行人和偏好；涉及实时信息时建议调用已授权工具。"},
    {"slug":"meeting-summary","name":"会议纪要","category":"通用办公","description":"将会议材料整理为结论、行动项、负责人和截止时间。","content":"# 会议纪要\n\n以结论、行动项、负责人、截止时间四部分输出；缺失信息明确标记待补充。"},
    {"slug":"web-design","name":"网页设计","category":"代码开发","description":"把用户需求转为信息架构、界面层级和可实施的前端建议。","content":"# 网页设计\n\n先给出页面目标、用户路径和组件清单，再输出可实施的视觉与交互建议。"},
    {"slug":"software-engineer","name":"本地软件工程","category":"代码开发","description":"在已选工作区内规划、修改、测试和审查代码，并坚持验收标准与最小修改范围。","content":"# 本地软件工程\n\n先读取工作区和相关代码，再给出简洁计划。需要修改时只调用已挂载的工作区工具；每次写入或测试均等待审批。修改后必须读取 diff 并运行适当测试。出现错误时依据真实日志定位原因，避免猜测。不得访问工作区外路径，不得声称未执行的测试已经通过。"},
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
    # 控制台展示用：direct / condition / loop / batch / intent。
    edge_type: str = "direct"


class DisconnectReq(BaseModel):
    source_id: str
    target_id: str
    conditional: bool = False


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
    arguments: Dict[str, Any] = Field(default_factory=dict)


class CreateWorkspaceReq(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    root_path: str
    read_only: bool = False
    allowed_commands: List[str] = Field(default_factory=list)
    create_if_missing: bool = False


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
    knowledge_base_bindings: List[Dict[str, Any]] = Field(default_factory=list)
    memory_bank_ids: List[str] = Field(default_factory=list)
    primary_memory_bank_id: Optional[str] = None
    memory_config: Dict[str, Any] = Field(default_factory=dict)
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
    knowledge_base_bindings: Optional[List[Dict[str, Any]]] = None
    memory_bank_ids: Optional[List[str]] = None
    primary_memory_bank_id: Optional[str] = None
    memory_config: Optional[Dict[str, Any]] = None
    prompt_variables: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class TestMCPReq(BaseModel):
    endpoint: str
    auth_type: str = "none"
    token: str = ""


class AddAgentMCPReq(BaseModel):
    name: str
    endpoint: str
    auth_type: str = "none"
    token: str = ""
    enabled_tools: List[str] = Field(default_factory=list)
    discovered_tools: List[Dict[str, Any]] = Field(default_factory=list)


class UpdateAgentMCPReq(BaseModel):
    enabled: Optional[bool] = None
    enabled_tools: Optional[List[str]] = None


class CreateApplicationRunReq(BaseModel):
    input: Dict[str, Any] = Field(default_factory=dict)
    recursion_limit: int = 50
    conversation_id: Optional[str] = None


class CreateMemoryBankReq(BaseModel):
    """创建控制台可挂载的记忆库资源。"""

    name: str
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateMemoryBankReq(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    retrieval_config: Optional[Dict[str, Any]] = None


class MemoryRuleReq(BaseModel):
    type: str = "fragment"
    name: str
    description: str = ""
    instruction: str
    source_types: List[str] = Field(default_factory=lambda:["user"])
    update_policy: str = "merge"
    retention_days: int = Field(180, ge=0, le=3650)
    target_scope: str = "project"
    enabled: bool = True


class CreateMemoryReq(BaseModel):
    content: Any
    scope: str = "project"
    scope_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateConsoleResourceReq(BaseModel):
    """创建组件、知识库、数据连接或治理资源。"""

    name: str
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeBaseReq(BaseModel):
    name: str = Field(max_length=80)
    description: str = ""
    workspace_id: str = "local"
    type: str = "document"
    edition: str = "standard"
    embedding_model: str = "hashing"
    retrieval_mode: str = "hybrid"
    chunk_strategy: str = "smart"
    chunk_size: int = 600
    chunk_overlap: int = 80
    similarity_threshold: float = .15
    top_k: int = 5
    rerank_enabled: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseUpdateReq(BaseModel):
    """Partial update payload; creation fields must not be required on PUT."""
    name: Optional[str] = Field(default=None, max_length=80)
    description: Optional[str] = None
    workspace_id: Optional[str] = None
    type: Optional[str] = None
    edition: Optional[str] = None
    embedding_model: Optional[str] = None
    retrieval_mode: Optional[str] = None
    chunk_strategy: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    similarity_threshold: Optional[float] = None
    top_k: Optional[int] = None
    rerank_enabled: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

class KnowledgeRetrieveReq(BaseModel):
    knowledge_base_ids: List[str] = Field(default_factory=list)
    query: str
    mode: str = "hybrid"
    top_k: int = 5
    threshold: float = .15
    labels: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    bindings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


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


def _approval_follow_up(result: Dict[str, Any]) -> str:
    """Create a safe, concise continuation visible after a tool decision."""
    name = str(result.get("display_name") or result.get("name") or "工具")
    if result.get("status") != "succeeded":
        return f"已批准 {name}，但调用失败：{result.get('error') or '未知错误'}"
    value = result.get("result")
    rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return f"已批准并完成 {name}。工具结果：{rendered[:1800]}"


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
    tool_connection_store: Optional[ToolConnectionStore] = None,
    application_store: Optional[ApplicationStore] = None,
    memory_bank_store: Optional[MemoryBankStore] = None,
    console_resource_store: Optional[ConsoleResourceStore] = None,
    api_key_store: Optional[ApiKeyStore] = None,
    api_audit_store: Optional[ApiAuditStore] = None,
    mcp_config_store: Optional[MCPConfigStore] = None,
    auth_store: Optional[AuthStore] = None,
    auth_required: bool = False,
    model_connection_store: Optional[ModelConnectionStore] = None,
    conversation_store: Optional[ConversationStore] = None,
    memory_audit_store: Optional[MemoryAuditStore] = None,
    workspace_store: Optional[WorkspaceStore] = None,
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
    tool_connections = tool_connection_store or ToolConnectionStore(
        os.environ.get("TOOL_CONNECTION_ROOT") or str(tools.root_dir.parent / "tool_connections")
    )
    mcp_oauth = MCPAuthorizationStore(os.environ.get("MCP_OAUTH_ROOT") or str(tool_connections.root_dir.parent / "mcp_oauth"))
    conversations = conversation_store or ConversationStore(os.environ.get("CONVERSATION_STORE_ROOT") or "runs/conversations")
    memory_audit = memory_audit_store or MemoryAuditStore(os.environ.get("MEMORY_AUDIT_ROOT") or "runs/memory_audit")
    memory_data_root = os.environ.get("MEMORY_DATA_ROOT") or str(memory_banks.root_dir.parent / "memory_data")
    console_resources = console_resource_store or ConsoleResourceStore(
        os.environ.get("CONSOLE_RESOURCE_STORE_ROOT") or "runs/console_resources"
    )
    knowledge = KnowledgeStore(os.environ.get("KNOWLEDGE_STORE_ROOT") or str(applications.root_dir.parent / "knowledge"))
    api_keys = api_key_store or ApiKeyStore(os.environ.get("API_KEY_STORE_ROOT") or "runs/api_keys")
    api_audit = api_audit_store or ApiAuditStore(
        os.environ.get("API_AUDIT_LOG_PATH") or "runs/audit/api_audit.jsonl"
    )
    mcp_configs = mcp_config_store or MCPConfigStore(os.environ.get("MCP_CONFIG_ROOT") or "runs/mcp")
    workspaces = workspace_store or WorkspaceStore(os.environ.get("WORKSPACE_STORE_ROOT") or str(applications.root_dir.parent / "workspaces"))
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
    runtime_factory = node_factory or AgentRuntimeFactory(
        tool_catalog_store=tools,
        model_connection_store=model_connections,
        mcp_config_store=mcp_configs,
        mcp_oauth_store=mcp_oauth,
        knowledge_store=knowledge,
        workspace_store=workspaces,
    )

    def _approval_requests(record: RunRecord) -> List[Dict[str, Any]]:
        return [event for event in record.events if event.get("type") == "approval_required"]

    def _unresolved_approvals(record: RunRecord) -> List[Dict[str, Any]]:
        decisions = dict(record.metadata.get("approval_decisions") or {})
        return [event for event in _approval_requests(record) if str(event.get("sequence")) not in decisions]

    def _approval_tool_context(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        try:
            metadata = dict(tools.get(str(tool_call.get("id") or "")).metadata)
        except KeyError:
            pass
        endpoint = str(metadata.get("mcp_url") or metadata.get("operation_url") or "")
        host = urllib.parse.urlparse(endpoint).hostname or ""
        service_name = str(
            metadata.get("connection_name")
            or metadata.get("mcp_name")
            or ("平台内置工具" if metadata.get("source") == "builtin" else host)
            or "外部工具服务"
        )
        return {
            "service_name": service_name,
            "service_host": host,
            "source": str(metadata.get("source") or "external"),
        }

    def _approval_scope(tool_call: Dict[str, Any]) -> str:
        """A task-scoped grant covers only bounded local edits and checks."""
        try:
            adapter = str(tools.get(str(tool_call.get("id") or "")).metadata.get("adapter") or "")
        except KeyError:
            adapter = str(tool_call.get("name") or "")
        return "workspace_engineering" if adapter in {
            "workspace_apply_patch", "workspace_write_files", "workspace_run_command",
        } else ""

    def _approval_prompt(node_name: Any, tool_call: Dict[str, Any], approval_context: Dict[str, str]) -> str:
        tool_name = tool_call.get("display_name") or tool_call.get("name") or "工具"
        if _approval_scope(tool_call) == "workspace_engineering":
            return (
                f"是否允许 {node_name} 在当前已选本地代码工作区执行本次工程任务？"
                f"批准后，本任务内的受限文件创建/修改和白名单本地检查会自动继续；"
                f"外部服务、恢复/删除等更高风险操作仍会单独请求批准。"
            )
        return f"是否允许 {node_name} 使用 {approval_context['service_name']} 的 {tool_name} 执行本次操作？"

    def _approval_waiting_text(record: RunRecord) -> str:
        pending = _unresolved_approvals(record)
        names = "、".join(
            str((event.get("tool_call") or {}).get("display_name") or (event.get("tool_call") or {}).get("name") or "工具")
            for event in pending
        )
        return f"任务已暂停，正在等待你批准以下工具调用：{names}。批准后我会使用真实结果继续完成任务；拒绝后我会说明未完成原因并收尾。"

    def _approval_spec(record: RunRecord, node_name: str):
        target = _orchestrator_for_run(record.workflow_id, record)
        return next((item for item in target.list_agents() if item.name == node_name), None)

    workflow_runtime_factory = WorkflowNodeRuntimeFactory(tools, model_connections, knowledge_store=knowledge, mcp_oauth_store=mcp_oauth, workspace_store=workspaces, skill_repository=skills, run_store=runs)
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
    # Protocol marker distinguishes an alive but stale backend from this build.
    runtime_info = {"run_stream_protocol": 2, "instance_id": __import__("uuid").uuid4().hex, "started_at": _utc_now(), "features": ["answer_delta", "tool_progress", "approval_resume"]}
    original_openapi = app.openapi

    def runtime_openapi():
        schema = original_openapi()
        schema["info"]["x-run-stream-protocol"] = runtime_info["run_stream_protocol"]
        return schema

    app.openapi = runtime_openapi

    @app.get("/api/system/runtime")
    def get_runtime_info() -> Dict[str, Any]:
        return dict(runtime_info)
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
            "/api/auth/register", "/api/auth/login", "/api/auth/forgot-password", "/api/auth/reset-password",
            # This is a one-time external OAuth redirect. Its state/PKCE
            # verification is the authentication boundary; requiring the
            # platform session here breaks localhost vs 127.0.0.1 callbacks.
            "/api/tool-connections/oauth/callback",
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

    def _model_selection_errors(app_record: ApplicationRecord, graph: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate fixed/AUTO selections against the shared model connection catalog."""
        errors: List[str] = []
        auto_status = model_connections.auto_status()

        def check(selection: str, label: str) -> None:
            # Empty is the legacy pre-model-management value. It keeps the old
            # development fallback until the application is explicitly saved as AUTO.
            if selection == "":
                return
            normalized = "auto" if selection in {"", "auto", "device", "edge", "cloud"} else selection
            if normalized == "auto":
                if not auto_status["ready"]:
                    missing = "、".join(name for tier, name in (("device", "端"), ("edge", "边"), ("cloud", "云")) if not auto_status["tiers"][tier]["ready"])
                    errors.append(f"{label}使用 AUTO，但{missing}模型尚未设置可用的默认连接")
                return
            try:
                connection = model_connections.get(normalized)
                if not connection.runnable:
                    errors.append(f"{label}选择的模型连接未启用、未配置或尚未测试成功")
            except KeyError:
                errors.append(f"{label}选择的模型连接不存在")

        if app_record.app_type == "agent":
            check(app_record.model, "智能体应用")
        elif graph is not None:
            for item in graph.get("agents", []):
                kind = str((item.get("config") or {}).get("node_kind") or "agent")
                config = item.get("config") or {}
                selection = str(config.get("model") or item.get("model") or "") if kind in {"task_planner", "result_aggregator", "quality_gate", "intent"} else str(item.get("model") or "")
                if kind not in {"agent", "llm", "task_planner", "result_aggregator", "quality_gate", "intent"}:
                    continue
                if kind == "result_aggregator" and str(config.get("mode") or "synthesize") == "ordered":
                    continue
                if selection:
                    check(selection, f"节点“{item.get('name')}”")
        return list(dict.fromkeys(errors))

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

    def _orchestrator_for_run(workflow_id: Optional[str], record: Optional[RunRecord] = None) -> Orchestrator:
        if workflow_id:
            target = Orchestrator.from_dict(workflows.get(workflow_id).graph)
        else:
            target = Orchestrator.from_dict(orch.to_dict())
        _apply_policy(target)
        application_id = str((record.metadata if record else {}).get("application_id") or "")
        if application_id:
            app_record = applications.get(application_id)
            if app_record.app_type == "workflow":
                workflow_tool_ids = [tool_id for tool_id in app_record.tool_ids if tools.exists(tool_id)]
                workflow_tool_set = set(workflow_tool_ids)
                for spec in target.list_agents():
                    kind = str(spec.config.get("node_kind") or "agent")
                    if kind in {"agent", "llm"}:
                        config = dict(spec.config)
                        node_tool_ids = [str(tool_id) for tool_id in (config.get("tool_ids") or [])]
                        inherited_tools = node_tool_ids or workflow_tool_ids
                        config["tool_ids"] = [tool_id for tool_id in inherited_tools if tool_id in workflow_tool_set]
                        target.update_agent(spec.id, config=config)
            bound = [bank_id for bank_id in app_record.memory_bank_ids if memory_banks.exists(bank_id)]
            primary = app_record.primary_memory_bank_id or (bound[0] if bound else None)
            if primary and primary in bound and normalize_memory_config(app_record.memory_config)["long_term_enabled"]:
                target.set_memory(MemoryBankRuntime(memory_data_root, primary, [bank_id for bank_id in bound if bank_id != primary]))
                memory_config=normalize_memory_config(app_record.memory_config)
                bank_config=normalize_memory_retrieval_config(memory_banks.get(primary).retrieval_config)
                target.set_memory_options(top_k=memory_config["memory_top_k"] if memory_config["retrieval_override_enabled"] else bank_config["top_k"], wakeup_level=memory_config["wakeup_level"])
        return target

    def _validate_application_workflow(graph: Dict[str, Any], *, runnable: bool = False, app_tool_ids: Optional[List[str]] = None) -> List[str]:
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
            # Conditional connections use ``<conditional>`` as a serialized
            # sentinel; their real targets are carried by ``path_map`` below.
            # It is not a node id and must not be reported as a dangling edge.
            if source not in ids or (target not in {"END", "<conditional>"} and target not in ids):
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
        allowed_tools = set(app_tool_ids or [])
        if app_tool_ids is not None:
            for item, kind in zip(agents, kinds):
                config = item.get("config") or {}
                if kind == "tool":
                    tool_id = str(config.get("tool_id") or "")
                    if tool_id and tool_id not in allowed_tools:
                        errors.append(f"工具节点 {item.get('name') or item.get('id')} 使用了未挂载到当前工作流的工具")
                if kind in {"agent", "llm"}:
                    invalid = [str(tool_id) for tool_id in (config.get("tool_ids") or []) if str(tool_id) not in allowed_tools]
                    if invalid:
                        errors.append(f"节点 {item.get('name') or item.get('id')} 包含未挂载到当前工作流的工具")
        if starts:
            adjacency: Dict[str, set[str]] = {node_id: set() for node_id in ids}
            contained_by_parent: Dict[str, set[str]] = {node_id: set() for node_id in ids}
            for edge in connections:
                source = str(edge.get("source") or "")
                targets = list((edge.get("path_map") or {}).values()) if edge.get("conditional") else [edge.get("target")]
                adjacency.setdefault(source, set()).update(str(target) for target in targets if target in ids)
            for item in agents:
                parent_id = str(item.get("parent_id") or "")
                if parent_id in ids:
                    contained_by_parent.setdefault(parent_id, set()).add(str(item.get("id")))
                for child_id in item.get("children") or []:
                    if str(child_id) in ids:
                        contained_by_parent.setdefault(str(item.get("id")), set()).add(str(child_id))
            reached, pending = set(), [str(starts[0].get("id"))]
            while pending:
                current_id = pending.pop()
                if current_id in reached:
                    continue
                reached.add(current_id)
                pending.extend(adjacency.get(current_id, set()) - reached)
                # Entering a team/container makes its members reachable even
                # though they intentionally have no static workflow edges.
                pending.extend(contained_by_parent.get(current_id, set()) - reached)
            if reached != ids:
                errors.append("所有节点必须能够从开始节点到达")
            if ends and str(ends[0].get("id")) not in reached:
                errors.append("结束节点必须能够从开始节点到达")
        for item, kind in zip(agents, kinds):
            config = item.get("config") or {}
            if kind in {"agent", "llm"} and not str(item.get("model") or "").strip():
                errors.append(f"节点“{item.get('name')}”必须单独选择模型")
            if kind == "tool" and not config.get("tool_id"):
                errors.append(f"工具节点“{item.get('name')}”尚未选择工具")
            if kind in {"condition", "intent", "loop", "batch", "task_planner", "quality_gate", "goal_gate", "recovery_boundary"} and not any(edge.get("source") == item.get("id") and edge.get("conditional") for edge in connections):
                errors.append(f"逻辑节点“{item.get('name')}”尚未配置分支连线")
            if kind in {"task_planner", "quality_gate"}:
                conditional = next((edge for edge in connections if edge.get("source") == item.get("id") and edge.get("conditional")), {})
                path_map = conditional.get("path_map") or {}
                required_routes = (
                    [str(config.get("execute_route") or "execute"), str(config.get("done_route") or "done"), str(config.get("failed_route") or "failed")]
                    if kind == "task_planner"
                    else [str(config.get("continue_route") or "continue")]
                )
                missing_routes = [route for route in required_routes if route not in path_map]
                if missing_routes:
                    errors.append(f"动态控制节点“{item.get('name')}”缺少出口：{'、'.join(missing_routes)}")
                if kind == "quality_gate":
                    feedback_target = str(path_map.get(str(config.get("continue_route") or "continue")) or "")
                    target_kind = next((str((candidate.get("config") or {}).get("node_kind") or "") for candidate in agents if str(candidate.get("id")) == feedback_target), "")
                    if feedback_target and target_kind != "task_planner":
                        errors.append(f"质量门“{item.get('name')}”的 TODO 反馈出口必须连接任务规划器")
            if kind == "intent" and not (config.get("model") or item.get("model")):
                errors.append(f"意图分类节点“{item.get('name')}”尚未选择模型")
            if kind == "task_planner" and not (config.get("model") or item.get("model")):
                errors.append(f"任务规划器“{item.get('name')}”尚未选择模型")
            if kind == "script" and config.get("code") and not config.get("output_field"):
                errors.append(f"脚本节点“{item.get('name')}”尚未配置输出字段")
            if kind in {"loop", "batch"}:
                arrays = config.get("input_arrays") or []
                if config.get("loop_type") == "array" or kind == "batch":
                    if not arrays or any(not value.get("path") or not value.get("item_field") for value in arrays):
                        errors.append(f"{item.get('name')}的输入数组配置不完整")
                child_ids = set(item.get("children") or [])
                if any(str((child.get("config") or {}).get("node_kind")) in {"loop", "batch"} for child in agents if child.get("id") in child_ids):
                    errors.append(f"{item.get('name')}内不能嵌套循环或批处理")
            if kind == "agent_team":
                child_ids = {str(value) for value in item.get("children") or []}
                members = [child for child in agents if str(child.get("id")) in child_ids and str((child.get("config") or {}).get("node_kind") or "agent") not in {"start", "end", "loop_start", "loop_end", "batch_start", "batch_end"}]
                if not members:
                    errors.append(f"动态智能体团队“{item.get('name')}”至少需要一个成员")
                supervisor_id = str(config.get("supervisor_id") or "")
                if supervisor_id and supervisor_id not in child_ids:
                    errors.append(f"动态智能体团队“{item.get('name')}”的主管必须是团队成员")
        return list(dict.fromkeys(errors))

    def _normalize_workflow_node_names(graph: Dict[str, Any]) -> Dict[str, Any]:
        labels = {"start":"开始","end":"结束","loop_start":"循环开始","loop_end":"迭代结束","batch_start":"批处理开始","batch_end":"批处理结束","llm":"大模型","knowledge":"知识库","tool":"工具","agent":"智能体","condition":"条件判断","intent":"意图分类","drift_guard":"防漂移检查","script":"脚本","assign":"变量赋值","loop":"循环","batch":"批处理"}
        counts: Dict[str, int] = {}
        agents = []
        for source in graph.get("agents", []):
            item = dict(source)
            kind = str((item.get("config") or {}).get("node_kind") or "agent")
            counts[kind] = counts.get(kind, 0) + 1
            item["name"] = f"{labels.get(kind, kind)}{counts[kind]}"
            agents.append(item)
        return {**graph, "agents": agents}

    def _append_event(record: RunRecord, event: Dict[str, Any]) -> Dict[str, Any]:
        enriched = {
            **event,
            "run_id": record.id,
            "sequence": len(record.events) + 1,
            "timestamp": _utc_now(),
        }
        record.events.append(enriched)
        runs.save(record)
        return enriched

    def _runtime_child_event_message(event: Dict[str, Any]) -> str:
        """Human-readable trace labels for nested batch and dynamic-team events."""
        labels = {
            "team_started": "动态团队开始评估可用成员",
            "candidate_filtered": "已完成资源硬过滤与多指标候选评分",
            "topology_reconfigured": "动态团队已重构本轮协作拓扑",
            "delegation_selected": "主管已动态委派子智能体",
            "handoff_selected": "主管已批准智能体自主 Handoff",
            "handoff_rejected": "主管拒绝 Handoff 请求",
            "team_member_started": "子智能体开始处理委派任务",
            "team_member_completed": "子智能体已返回委派结果",
            "goal_checked": "团队已完成本轮目标检查",
            "team_member_failed": "子智能体执行失败，正在评估替代成员",
            "fallback_selected": "已选择备用子智能体接管任务",
            "team_aggregated": "动态团队已汇聚成员结果",
        }
        if event.get("type") in labels:
            parts = [labels[event["type"]]]
            if event.get("source") and event.get("target"):
                parts.append(f"：{event['source']} → {event['target']}")
            elif event.get("node"):
                parts.append(f"：{event['node']}")
            return "".join(parts)
        return (
            f"{event.get('node')} 已完成批处理项"
            if event.get("type") == "node_end"
            else f"进入 {event.get('node')}，开始处理批处理项"
        )

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

    def _memory_binding(ids: List[str], primary: Optional[str]) -> tuple[List[str], Optional[str]]:
        bound = list(dict.fromkeys(str(item) for item in ids if str(item)))
        effective_primary = primary or (bound[0] if bound else None)
        if effective_primary and effective_primary not in bound:
            raise ValueError("主记忆库必须同时包含在已绑定记忆库中")
        return bound, effective_primary

    def _scope_counts(items: List[Any]) -> Dict[str, int]:
        counts = {scope.value: 0 for scope in MemoryScope.hierarchy()}
        for item in items:
            key = item.scope.value
            counts[key] = counts.get(key, 0) + 1
        return counts

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
        if any(str((item.get("config") or {}).get("node_kind")) == "task_planner" for item in target.to_dict().get("agents", [])):
            record.input = {**record.input, "original_goal": _task_text(record.input)}
            record.metadata = {**record.metadata, "todos": [], "plan_status": "waiting_for_planner", "runtime_planner": True}
            _append_event(record, {"type": "plan_waiting", "message": "等待任务规划器生成可执行 TODO"})
            return
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

    def _rule_memory_candidates(app_record: ApplicationRecord, source_text: str, answer_text: str, rules: List[Dict[str,Any]]) -> tuple[List[Dict[str,str]],str]:
        if not app_record.model or app_record.model in {"auto","device","edge","cloud"}:
            return [],"应用未选择已测试的真实模型连接"
        try:
            connection=model_connections.get(app_record.model)
            if not connection.enabled or connection.test_status!="succeeded" or not connection.to_dict()["configured"]:
                return [],"模型连接未配置、未启用或尚未测试成功"
            rule_text="\n".join(f"- {item['id']} | {item['name']}: {item['instruction']}" for item in rules if item.get("enabled"))
            prompt=f"""你是长期记忆提取器。只提取对未来仍有价值且符合规则的信息，禁止密码、密钥、令牌和敏感身份属性。\n规则：\n{rule_text}\n用户消息：{source_text}\n助手回复：{answer_text}\n只返回 JSON 对象 {{\"items\":[{{\"rule_id\":\"...\",\"content\":\"...\"}}]}}；没有候选时 items 为空数组。"""
            headers={"Content-Type":"application/json","Accept":"application/json"}
            if connection.api_key_env and os.environ.get(connection.api_key_env):headers["Authorization"]=f"Bearer {os.environ[connection.api_key_env]}"
            body=json.dumps({"model":connection.model_id,"messages":[{"role":"user","content":prompt}],"temperature":0,"response_format":{"type":"json_object"}}).encode("utf-8")
            req=urllib.request.Request(f"{connection.base_url}/chat/completions",data=body,headers=headers,method="POST")
            with urllib.request.urlopen(req,timeout=60) as response:payload=json.loads(response.read().decode("utf-8"))
            raw=str((((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")).strip().removeprefix("```json").removesuffix("```").strip()
            parsed=json.loads(raw);parsed=parsed.get("items",[]) if isinstance(parsed,dict) else parsed
            if not isinstance(parsed,list):raise ValueError("模型输出不是数组")
            allowed={str(item.get("id")) for item in rules if item.get("enabled")}
            return [{"rule_id":str(item.get("rule_id") or ""),"content":str(item.get("content") or "").strip()} for item in parsed if isinstance(item,dict) and str(item.get("rule_id")) in allowed and str(item.get("content") or "").strip()],""
        except Exception as exc:
            return [],str(exc)

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
            target = _orchestrator_for_run(record.workflow_id, record)
            _seed_todo_events(record, target)
            is_visual_workflow = any(
                (item.get("config") or {}).get("node_kind")
                for item in target.to_dict().get("agents", [])
            )
            visual_runtime = WorkflowNodeRuntimeFactory(tools, model_connections, target.to_dict(), knowledge_store=knowledge, mcp_oauth_store=mcp_oauth, workspace_store=workspaces, skill_repository=skills, run_store=runs) if is_visual_workflow else None
            compiled = target.build_graph(
                node_factory=visual_runtime if visual_runtime is not None else runtime_factory,
                recursion_limit=record.recursion_limit,
            )
            run_input = {
                **record.input,
                "run_id": record.id,
                "__run_id__": record.id,
                "__workflow_id__": record.workflow_id,
                "__application_id__": str(record.metadata.get("application_id") or ""),
                "__owner_user_id__": str(record.metadata.get("owner_user_id") or "local-user"),
                "task_id": record.id,
                "project_id": str(record.metadata.get("application_id") or record.workflow_id or "default-project"),
                "global_id": str(record.metadata.get("owner_user_id") or "default"),
            }
            from ..modules.live_events import live_events
            async for event in live_events(compiled.astream(
                run_input,
                record.recursion_limit,
                run_id=record.id,
            ), cancel_check=lambda: runs.get(record.id).status == "cancel_requested"):
                child_events: List[Dict[str, Any]] = []
                if event.get("type") == "node_start":
                    record.metadata["active_agent"] = event.get("node")
                    event["message"] = f"进入 {event.get('node')}，开始处理当前步骤"
                elif event.get("type") == "node_end":
                    update = dict(event.get("update") or {})
                    child_events = [
                        *list(update.pop("__runtime_child_events__", []) or []),
                        *list(update.pop("__runtime_team_events__", []) or []),
                    ]
                    event["update"] = update
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
                    if event.get("type") == "tool_finished":
                        _append_event(record, event)
                    record.status = "canceled"
                    record.canceled_at = latest.canceled_at or _utc_now()
                    record.finished_at = record.canceled_at
                    record.metadata = latest.metadata
                    runs.save(record)
                    return
                _append_event(record, event)
                if event.get("type") in {"model_started", "model_finished", "model_failed", "answer_delta", "answer_mode", "tool_started", "tool_finished", "skill_applied", "knowledge_retrieval_start", "knowledge_retrieval_end", "progress_summary"}:
                    continue
                if event.get("type") == "node_end":
                    for child_event in child_events:
                        child_event["message"] = child_event.get("message") or (
                            _runtime_child_event_message(child_event)
                        )
                        _append_event(record, child_event)
                    for tool_call in event.get("tool_calls") or []:
                        approval_context = _approval_tool_context(tool_call)
                        if tool_call.get("status") == "approval_required":
                            tool_call = {**tool_call, **approval_context}
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
                                "approval_prompt": _approval_prompt(event.get("node"), tool_call, approval_context)
                                if tool_call.get("status") == "approval_required"
                                else "",
                                "message": (
                                    f"{event.get('node')} 请求审批工具 {tool_call.get('display_name') or tool_call.get('name')}"
                                    if tool_call.get("status") == "approval_required"
                                    else f"{event.get('node')} 已调用工具 {tool_call.get('display_name') or tool_call.get('name')}"
                                ),
                            },
                        )
                    ledger = dict((event.get("update") or {}).get("plan_ledger") or {})
                    if ledger:
                        record.metadata["todos"] = list(ledger.get("todos") or [])
                        record.metadata["plan_status"] = str(ledger.get("status") or "running")
                        record.metadata["active_todo_id"] = str(ledger.get("current_todo_id") or "")
                        _append_event(record, {"type": "todo_plan_updated", "node": event.get("node"), "message": f"{event.get('node')} 已更新可执行 TODO 计划", "todos": record.metadata["todos"], "active_todo_id": record.metadata["active_todo_id"], "plan_status": record.metadata["plan_status"]})
                    elif not record.metadata.get("runtime_planner"):
                        _advance_todo(record, node=str(event.get("node") or ""), output=str(event.get("output") or ""))
                    quality_report = dict((event.get("update") or {}).get("quality_report") or {})
                    if quality_report:
                        _append_event(record, {"type": "quality_gate_decision", "node": event.get("node"), "message": f"质量门判定：{quality_report.get('decision')}", "decision": quality_report.get("decision"), "score": quality_report.get("overall_score"), "issues": quality_report.get("issues") or [], "todo_id": quality_report.get("todo_id")})
                if event.get("type") == "final":
                    record.state = dict(event.get("state") or {})
                runs.save(record)
            pending_approvals = _unresolved_approvals(record)
            if pending_approvals:
                waiting_text = _approval_waiting_text(record)
                record.status = "waiting_approval"
                record.finished_at = None
                record.state = {
                    **dict(record.state or {}),
                    "input": waiting_text,
                    "approval_follow_up": waiting_text,
                }
                record.metadata = {
                    **record.metadata,
                    "active_agent": None,
                    "duration_ms": round((time.perf_counter() - started_clock) * 1000, 2),
                    "pending_approval_sequences": [int(item.get("sequence") or 0) for item in pending_approvals],
                    "summary": _build_run_summary(record),
                }
                record.metadata["summary"]["title"] = "等待工具审批"
                record.metadata["summary"]["final_output"] = waiting_text
                runs.save(record)
                return
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
            summary = record.metadata["summary"]
            conversation_id = record.metadata.get("conversation_id")
            if conversation_id:
                try:
                    conversation = conversations.get(str(conversation_id))
                    final_output = str(summary.get("final_output") or "")
                    if final_output:
                        conversation.messages.append({"role":"assistant","content":final_output,"created_at":_utc_now(),"run_id":record.id})
                        conversations.save(conversation)
                except KeyError:
                    pass
            app_id = str(record.metadata.get("application_id") or "")
            app_record = applications.get(app_id) if app_id else None
            if app_record:
                memory_config = normalize_memory_config(app_record.memory_config)
                actions = []
                source_text = str(record.input.get("input") or "")
                if memory_config["long_term_enabled"] and memory_config["auto_write"] and app_record.primary_memory_bank_id:
                    primary_bank=memory_banks.get(app_record.primary_memory_bank_id)
                    enabled_rules=[item for item in primary_bank.metadata.get("rules",[]) if item.get("enabled")]
                    candidates,extract_error=_rule_memory_candidates(app_record,source_text,str(summary.get("final_output") or ""),enabled_rules)
                    if extract_error:
                        actions.append(memory_audit.append(app_record.id,{"action":"pending","reason":"model_extraction_failed","failure_reason":extract_error,"model_connection_id":app_record.model,"run_id":record.id,"conversation_id":conversation_id,"bank_id":app_record.primary_memory_bank_id}))
                    store = HybridTieredMemoryStore(os.path.join(memory_data_root, app_record.primary_memory_bank_id))
                    try:
                        existing = list_bank_memories(memory_data_root, app_record.primary_memory_bank_id, limit=500)
                        existing_text = {str(item.content).strip().casefold() for item in existing}
                        for extracted in candidates:
                            candidate,rule_id=extracted["content"],extracted["rule_id"]
                            matched_rule=next((item for item in enabled_rules if item.get("id")==rule_id),{})
                            retention_days=int(matched_rule.get("retention_days") or 0)
                            if memory_config["sensitive_filter"] and SENSITIVE_PATTERN.search(candidate):
                                action = {"action":"blocked","reason":"sensitive_information","candidate":candidate}
                            elif memory_config["deduplicate"] and candidate.strip().casefold() in existing_text:
                                action = {"action":"ignored","reason":"duplicate","candidate":candidate}
                            else:
                                item = store.append(candidate, MemoryScope.PROJECT, context=MemoryContext(project_id=app_record.id, global_id=app_record.owner_user_id or "default"), tags=["automatic"], source_run_id=record.id, source_conversation_id=conversation_id, rule_id=rule_id, model_connection_id=app_record.model, expires_at=(datetime.now(timezone.utc)+timedelta(days=retention_days)).isoformat() if retention_days else None)
                                action = {"action":"added","reason":"durable_project_memory","candidate":candidate,"memory_id":item.id,"scope":"project"}
                                existing_text.add(candidate.strip().casefold())
                            action = memory_audit.append(app_record.id, {**action,"rule_id":rule_id,"model_connection_id":app_record.model,"run_id":record.id,"conversation_id":conversation_id,"bank_id":app_record.primary_memory_bank_id})
                            actions.append(action)
                    finally:
                        store.close()
                record.metadata["memory_actions"] = actions
            runs.save(record)
        except Exception as e:  # noqa: BLE001 - API persists failures for polling
            canceled = runs.get(record.id).status in {"cancel_requested", "canceled"}
            record.status = "canceled" if canceled else "failed"
            record.error = None if canceled else str(e)
            record.finished_at = _utc_now()
            record.metadata = {
                **record.metadata,
                "active_agent": None,
                "duration_ms": round((time.perf_counter() - started_clock) * 1000, 2),
                "summary": _build_run_summary(record),
            }
            runs.save(record)
        finally:
            target_or_none = locals().get("target")
            memory_or_none = getattr(target_or_none, "_memory", None)
            if memory_or_none is not None:
                memory_or_none.close()

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

    @app.post("/api/mcp/test")
    async def test_mcp(req: TestMCPReq) -> Dict[str, Any]:
        try:
            return await test_mcp_connection(
                endpoint=req.endpoint,
                auth_type=req.auth_type,
                token=req.token,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        except Exception as e:  # noqa: BLE001 - show remote MCP failures to UI
            raise HTTPException(status_code=502, detail=f"MCP 连接失败：{e}")

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

    @app.get("/api/agents/{agent_id}/mcp")
    def list_agent_mcp(agent_id: str) -> Dict[str, Any]:
        try:
            orch.get_agent(agent_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"items": mcp_configs.agent_payload(agent_id)}

    @app.post("/api/agents/{agent_id}/mcp")
    async def add_agent_mcp(agent_id: str, req: AddAgentMCPReq, request: Request) -> Dict[str, Any]:
        try:
            orch.get_agent(agent_id)
            tools_payload = req.discovered_tools
            if not tools_payload:
                discovered = await list_mcp_tools(
                    endpoint=req.endpoint,
                    auth_type=req.auth_type,
                    token=req.token,
                )
            else:
                discovered = [MCPToolRecord.from_dict(item) for item in tools_payload]
            enabled_tools = req.enabled_tools or [tool.name for tool in discovered]
            user = getattr(request.state, "user", None)
            server = mcp_configs.create_server(
                name=req.name,
                endpoint=req.endpoint,
                auth_type=req.auth_type,
                token=req.token,
                tools=discovered,
                enabled_tools=enabled_tools,
                user_id=getattr(user, "id", "") if user else "",
            )
            binding = mcp_configs.bind_agent(
                agent_id=agent_id,
                mcp_server_id=server.id,
                enabled_tools=enabled_tools,
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"MCP 保存失败：{e}")
        return {
            "ok": True,
            "item": {
                **binding.to_dict(),
                "server": server.to_dict(),
                "tools": [
                    {**tool.to_dict(), "enabled_for_agent": tool.name in set(binding.enabled_tools)}
                    for tool in server.tools
                ],
            },
        }

    @app.put("/api/agents/{agent_id}/mcp/{binding_id}")
    def update_agent_mcp(agent_id: str, binding_id: str, req: UpdateAgentMCPReq) -> Dict[str, Any]:
        try:
            orch.get_agent(agent_id)
            binding = mcp_configs.get_binding(binding_id)
            if binding.agent_id != agent_id:
                raise KeyError(f"agent mcp binding {binding_id!r} not found")
            if req.enabled is not None:
                binding.enabled = req.enabled
            if req.enabled_tools is not None:
                binding.enabled_tools = req.enabled_tools
            binding = mcp_configs.save_binding(binding)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"ok": True, "item": mcp_configs.agent_payload(agent_id)}

    @app.post("/api/agents/{agent_id}/mcp/{binding_id}/sync")
    async def sync_agent_mcp(agent_id: str, binding_id: str) -> Dict[str, Any]:
        try:
            orch.get_agent(agent_id)
            binding = mcp_configs.get_binding(binding_id)
            if binding.agent_id != agent_id:
                raise KeyError(f"agent mcp binding {binding_id!r} not found")
            server = mcp_configs.get_server(binding.mcp_server_id)
            token = mcp_configs.decrypt_secret(server.auth_secret)
            discovered = await list_mcp_tools(
                endpoint=server.endpoint,
                auth_type=server.auth_type,
                token=token,
            )
            existing_enabled = set(binding.enabled_tools)
            server.tools = [
                MCPToolRecord(
                    name=tool.name,
                    title=tool.title,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    enabled=tool.name in existing_enabled or not existing_enabled,
                )
                for tool in discovered
            ]
            server.last_synced_at = _utc_now()
            server.status = "active"
            server = mcp_configs.save_server(server)
            binding.enabled_tools = [tool.name for tool in server.tools if tool.enabled]
            binding = mcp_configs.save_binding(binding)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"MCP 同步失败：{e}")
        return {"ok": True, "item": mcp_configs.agent_payload(agent_id)}

    @app.delete("/api/agents/{agent_id}/mcp/{binding_id}")
    def delete_agent_mcp(agent_id: str, binding_id: str) -> Dict[str, Any]:
        try:
            orch.get_agent(agent_id)
            binding = mcp_configs.get_binding(binding_id)
            if binding.agent_id != agent_id:
                raise KeyError(f"agent mcp binding {binding_id!r} not found")
            mcp_configs.delete_server(binding.mcp_server_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"ok": True}

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
                orch.connect_conditional(
                    req.source_id,
                    req.condition_key,
                    path_map,
                    edge_type=req.edge_type or "condition",
                )
            else:
                orch.connect(req.source_id, _resolve_target(req.target_id), edge_type=req.edge_type)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"ok": True}

    @app.delete("/api/connections")
    def disconnect(req: DisconnectReq) -> Dict[str, Any]:
        orch.disconnect(req.source_id, _resolve_target(req.target_id), conditional=req.conditional)
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

    @app.get("/api/runs/{run_id}/files/preview")
    def get_run_file_preview(run_id: str, request: Request, response: Response, workspace_id: str, path: str) -> Dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        try:
            record = _owned_run(run_id, request)
            # Legacy ownerless runs must not grant access to current local files.
            if not record.metadata.get("owner_user_id"):
                raise HTTPException(status_code=403, detail="旧运行缺少归属信息，不能读取当前工作区文件")
            return preview_run_file(record.events, workspaces, workspace_id, path)
        except (KeyError, FileNotFoundError):
            raise HTTPException(status_code=404, detail="运行、工作区或文件不存在；文件可能已删除或移动")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except OSError:
            raise HTTPException(status_code=400, detail="无法读取文件，请检查文件权限或占用情况")

    @app.get("/api/runs/{run_id}/events")
    async def stream_run_events(run_id: str, request: Request, after: int = 0):
        """用 SSE 推送持久化运行事件；断线后可通过 after 继续。"""
        try:
            _owned_run(run_id, request)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

        async def _event_stream():
            try:
                cursor = max(0, after, int(request.headers.get("last-event-id") or 0))
            except ValueError:
                cursor = max(0, after)
            idle_ticks = 0
            while True:
                if await request.is_disconnected():
                    break
                record = runs.get(run_id)
                while cursor < len(record.events):
                    event = record.events[cursor]
                    cursor += 1
                    yield f"id: {cursor}\nevent: run_event\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    idle_ticks = 0
                if record.status in {"succeeded", "failed", "canceled", "waiting_approval"}:
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
                await asyncio.sleep(0.15)

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

    approval_locks = {}

    def _record_approval_decision(run_id: str, sequence: int, *, approved: bool, reason: str = "") -> Dict[str, Any]:
        import threading
        from ..modules.live_events import sink, node_name, cancel_signal
        lock = approval_locks.setdefault(run_id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="该运行正在处理审批，请勿重复提交")
        try:
            record = runs.get(run_id)
            if record.status in {"running", "cancel_requested", "canceled"}:
                raise HTTPException(status_code=409, detail="该运行正在执行或已取消，不能重复审批")
            token = sink.set(lambda event: _append_event(record, event))
            node_token = node_name.set(str(next((e.get("node") for e in record.events if e.get("sequence") == sequence), "")))
            class ApprovalCancellation:
                def is_set(self):
                    return runs.get(run_id).status in {"cancel_requested", "canceled"}
            cancel_token = cancel_signal.set(ApprovalCancellation())
            try:
                return _apply_approval_decision(record, sequence, approved=approved, reason=reason)
            finally:
                sink.reset(token)
                node_name.reset(node_token)
                cancel_signal.reset(cancel_token)
        finally:
            lock.release()

    def _apply_approval_decision(record: RunRecord, sequence: int, *, approved: bool, reason: str = "") -> Dict[str, Any]:
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
        existing_decision = decisions.get(str(sequence))
        # An approval can have completed its side effect just before a model
        # continuation fails.  Treat an identical retry as a safe resume:
        # never execute the tool twice, only rebuild the continuation from its
        # recorded result.  This also makes browser retries idempotent.
        replay_completed_decision = existing_decision is not None
        if replay_completed_decision:
            if bool(existing_decision.get("approved")) != approved:
                raise HTTPException(status_code=409, detail="该工具调用已经完成相反的审批决定")
            result_event = next(
                (
                    event for event in reversed(record.events)
                    if event.get("type") == "tool_result" and int(event.get("approval_sequence") or 0) == sequence
                ),
                None,
            )
            if result_event is None:
                raise HTTPException(status_code=409, detail="该工具调用正在处理，请稍后重试")
            result = dict(result_event.get("tool_call") or {})
        else:
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
        record.status = "running"
        record.finished_at = None
        record.state = {**dict(record.state or {}), "approval_follow_up": "审批已处理，正在继续执行…"}
        record.metadata["pending_approval_sequences"] = [int(item.get("sequence") or 0) for item in _unresolved_approvals(record)]
        runs.save(record)
        _append_event(record, {"type": "approval_resumed", "node": target.get("node"), "approval_sequence": sequence, "message": "已批准，正在执行工具并继续任务" if approved else "已拒绝该工具，正在整理结果"})
        tool_call = dict(target.get("tool_call") or {})
        if not replay_completed_decision and approved:
            try:
                tool_id = str(tool_call.get("id") or "")
                try:
                    tool = tools.get(tool_id)
                except KeyError:
                    spec = _approval_spec(record, str(target.get("node") or ""))
                    dynamic = mcp_configs.runtime_tools_for_agent(spec.id) if spec is not None else []
                    payload = next((item for item in dynamic if str(item.get("id")) == tool_id), None)
                    if payload is None:
                        raise KeyError(f"tool {tool_id!r} not found")
                    tool = ToolRecord.from_dict(payload)
                call_arguments = dict(tool_call.get("arguments") or {})
                task_text = str(call_arguments.get("task") or record.input.get("input") or "")
                result = ToolRuntime(tools, mcp_oauth, workspaces).execute(
                    tool, task_text, arguments=call_arguments, bypass_approval=True
                ).to_dict()
            except Exception as exc:  # noqa: BLE001 - 审批后的执行错误必须留在审计轨迹中
                result = {**tool_call, "status": "failed", "error": str(exc)}
            _append_event(
                record,
                {
                    "type": "tool_result",
                    "node": target.get("node"),
                    "approval_sequence": sequence,
                    "tool_call": result,
                    "message": f"审批后已执行工具 {result.get('display_name') or result.get('name')}",
                },
            )
        elif not replay_completed_decision:
            result = {**tool_call, "status": "rejected", "error": reason or "用户拒绝执行"}
            _append_event(
                record,
                {
                    "type": "tool_result",
                    "node": target.get("node"),
                    "approval_sequence": sequence,
                    "tool_call": result,
                    "message": "工具调用已被用户拒绝",
                },
            )

        if approved and result.get("status") == "succeeded":
            scope = _approval_scope(tool_call)
            if scope:
                state = dict(record.state or {})
                state["__approved_tool_scopes__"] = sorted(set(state.get("__approved_tool_scopes__") or []) | {scope})
                record.state = state

        pending = _unresolved_approvals(record)
        if pending:
            follow_up = _approval_waiting_text(record)
            record.status = "waiting_approval"
            record.metadata["pending_approval_sequences"] = [int(item.get("sequence") or 0) for item in pending]
            record.state = {**dict(record.state or {}), "input": follow_up, "approval_follow_up": follow_up}
            record.metadata["summary"] = _build_run_summary(record)
            record.metadata["summary"]["title"] = "等待工具审批"
            record.metadata["summary"]["final_output"] = follow_up
            runs.save(record)
            return record.to_dict()

        outcomes: List[Dict[str, Any]] = []
        for request_event in _approval_requests(record):
            request_sequence = int(request_event.get("sequence") or 0)
            decision = decisions.get(str(request_sequence)) or {}
            result_event = next(
                (
                    event for event in reversed(record.events)
                    if event.get("type") == "tool_result" and int(event.get("approval_sequence") or 0) == request_sequence
                ),
                {},
            )
            outcomes.append(
                {
                    "approved": bool(decision.get("approved")),
                    "reason": str(decision.get("reason") or ""),
                    "tool_call": dict(request_event.get("tool_call") or {}),
                    "result": dict(result_event.get("tool_call") or {}),
                }
            )
        spec = _approval_spec(record, str(target.get("node") or ""))
        task_text = str(record.input.get("original_goal") or record.input.get("input") or "")
        continuation_tool_calls: List[Dict[str, Any]] = []
        if isinstance(runtime_factory, AgentRuntimeFactory) and spec is not None:
            continuation, continuation_tool_calls = runtime_factory.continue_tool_loop_after_tool_decisions(
                spec,
                task_text=task_text,
                outcomes=outcomes,
                state=record.state,
            )
            follow_up = continuation.text
            continuation_meta = continuation.to_dict()
        else:
            rejected = [item for item in outcomes if not item.get("approved")]
            follow_up = (
                f"因为当前所需工具未得到批准，所以相关操作没有执行。你可以重新发起并批准，或让我改用其他方式。"
                if rejected
                else _approval_follow_up(result)
            )
            continuation_meta = {"executor": "ApprovalContinuation", "model": "", "metadata": {"fallback": True}}

        if not continuation_meta.get("success", True):
            record.status = "failed"
            record.finished_at = _utc_now()
            record.error = str(continuation_meta.get("error") or "审批后的模型续跑失败")
            record.state = {**dict(record.state or {}), "input": follow_up, "approval_follow_up": follow_up}
            _append_event(record, {
                "type": "approval_continuation_failed",
                "node": target.get("node"),
                "message": follow_up,
                "error": record.error,
                "retryable": bool(continuation_meta.get("retryable")),
            })
            record.metadata["summary"] = _build_run_summary(record)
            runs.save(record)
            return record.to_dict()

        for tool_call in continuation_tool_calls:
            approval_context = _approval_tool_context(tool_call)
            if tool_call.get("status") == "approval_required":
                tool_call = {**tool_call, **approval_context}
            _append_event(
                record,
                {
                    "type": "approval_required" if tool_call.get("status") == "approval_required" else "tool_result",
                    "node": target.get("node"),
                    "tool_call": tool_call,
                    "approval_prompt": _approval_prompt(target.get("node"), tool_call, approval_context)
                    if tool_call.get("status") == "approval_required"
                    else "",
                    "message": (
                        f"{target.get('node')} 请求审批工具 {tool_call.get('display_name') or tool_call.get('name')}"
                        if tool_call.get("status") == "approval_required"
                        else f"{target.get('node')} 已调用工具 {tool_call.get('display_name') or tool_call.get('name')}"
                    ),
                },
            )

        pending_after_continuation = _unresolved_approvals(record)
        if pending_after_continuation:
            waiting_text = _approval_waiting_text(record)
            record.status = "waiting_approval"
            record.finished_at = None
            record.error = None
            record.state = {**dict(record.state or {}), "input": waiting_text, "approval_follow_up": waiting_text}
            record.metadata["approval_outcome"] = "approved"
            record.metadata["pending_approval_sequences"] = [int(item.get("sequence") or 0) for item in pending_after_continuation]
            record.metadata["summary"] = _build_run_summary(record)
            record.metadata["summary"]["title"] = "等待工具审批"
            record.metadata["summary"]["final_output"] = waiting_text
            runs.save(record)
            return record.to_dict()

        record.status = "succeeded"
        record.finished_at = _utc_now()
        record.error = None
        record.state = {**dict(record.state or {}), "input": follow_up, "approval_follow_up": follow_up}
        messages = list(record.state.get("messages") or [])
        messages.append({"agent": str(target.get("node") or "工具审批"), "content": follow_up, "runtime": "approval_continuation", "result": continuation_meta})
        record.state["messages"] = messages
        rejected_any = any(not item.get("approved") for item in outcomes)
        record.metadata["approval_outcome"] = "rejected" if rejected_any else "approved"
        record.metadata["pending_approval_sequences"] = []
        _append_event(record, {
            "type": "approval_continuation",
            "node": target.get("node"),
            "message": follow_up,
            "executor": continuation_meta.get("executor"),
            "model": continuation_meta.get("model"),
            "approval_outcome": record.metadata["approval_outcome"],
        })
        record.metadata["summary"] = _build_run_summary(record)
        conversation_id = str(record.metadata.get("conversation_id") or "")
        if conversation_id:
            try:
                conversation = conversations.get(conversation_id)
                conversation.messages.append({"role": "assistant", "content": follow_up, "created_at": _utc_now(), "run_id": record.id})
                conversations.save(conversation)
            except KeyError:
                pass
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

    @app.get("/api/model-connections/auto-status")
    def get_model_auto_status() -> Dict[str, Any]:
        return model_connections.auto_status()

    @app.post("/api/model-connections")
    def create_model_connection(req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return model_connections.create(req).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/model-connections/auto-routing/{tier}")
    def update_model_auto_routing(tier: str, req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            connection_id = str(req.get("connection_id") or "").strip() or None
            model_connections.assign_auto_tier(tier, connection_id)
            return model_connections.auto_status()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/model-connections/{connection_id}")
    def update_model_connection(connection_id: str, req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            current = model_connections.get(connection_id)
            if req.get("enabled") is False:
                if current.auto_default:
                    raise ValueError("请先替换或取消该连接的 AUTO 默认配置，再停用")
                referenced = any(item.model == connection_id for item in applications.list())
                referenced = referenced or any(
                    str(node.get("model") or "") == connection_id
                    for workflow in workflows.list()
                    for node in workflow.graph.get("agents", [])
                )
                if referenced:
                    raise ValueError("该模型仍被应用或工作流引用，请先替换模型后再停用")
            # Merge from storage so an omitted write-only api_key is retained.
            return model_connections.save(ModelConnection.from_dict({**current.to_storage_dict(), **req, "id": connection_id})).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/model-connections/{connection_id}/test")
    def test_model_connection(connection_id: str) -> Dict[str, Any]:
        try:
            test_started=time.perf_counter()
            item = model_connections.get(connection_id)
            if not item.base_url:
                raise ValueError("模型连接缺少 base_url")
            headers = {"Accept": "application/json"}
            api_key = connection_api_key(item)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            request = urllib.request.Request(f"{item.base_url}/models", headers=headers)
            with urllib.request.urlopen(request, timeout=8) as response:  # noqa: S310 - URL is administrator configured
                if response.status >= 400: raise ValueError(f"模型服务返回 HTTP {response.status}")
            # Connection tests should verify credentials and basic inference,
            # not spend the entire tiny output budget on a reasoning trace.
            chat_body=json.dumps({"model":item.model_id,"messages":[{"role":"user","content":"Reply with exactly: OK"}],"thinking":{"type":"disabled"},"temperature":0,"max_tokens":32}).encode("utf-8")
            chat_headers={**headers,"Content-Type":"application/json"}
            chat_request=urllib.request.Request(f"{item.base_url}/chat/completions",data=chat_body,headers=chat_headers,method="POST")
            with urllib.request.urlopen(chat_request,timeout=30) as response:  # noqa: S310 - administrator configured
                payload=json.loads(response.read().decode("utf-8"))
            answer=str((((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")).strip()
            if not answer:raise ValueError("模型推理未返回文本")
            item.test_status = "succeeded"
            saved=model_connections.save(item).to_dict()
            return {**saved,"test_result":{"model":item.model_id,"latency_ms":round((time.perf_counter()-test_started)*1000,2),"response":answer[:120],"real_inference":True}}
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
            memory_ids, primary_memory_id = _memory_binding(req.memory_bank_ids, req.primary_memory_bank_id)
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
                model="" if req.app_type == "workflow" else req.model,
                system_prompt=req.system_prompt,
                avatar_url=req.avatar_url,
                tool_ids=req.tool_ids,
                skill_ids=req.skill_ids,
                knowledge_base_ids=req.knowledge_base_ids,
                knowledge_base_bindings=req.knowledge_base_bindings,
                memory_bank_ids=memory_ids,
                primary_memory_bank_id=primary_memory_id,
                memory_config=normalize_memory_config(req.memory_config),
                prompt_variables=req.prompt_variables,
                owner_user_id=_request_user_id(request),
                metadata=req.metadata,
            )
            draft = Orchestrator()
            if req.app_type == "workflow":
                entry_id = draft.create_agent(
                    name="开始1", description="接收工作流输入", config={"node_kind": "start", "input_fields": ["input"]}
                )
                end_id = draft.create_agent(
                    name="结束1", description="返回工作流最终输出", config={"node_kind": "end", "output_field": "input"}
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
                        "knowledge_base_bindings": req.knowledge_base_bindings,
                        "memory_bank_ids": memory_ids,
                        "primary_memory_bank_id": primary_memory_id,
                        "memory_config": normalize_memory_config(req.memory_config),
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
            requested_memory_ids = req.memory_bank_ids if req.memory_bank_ids is not None else current.memory_bank_ids
            primary_was_supplied = "primary_memory_bank_id" in req.model_fields_set
            requested_primary = req.primary_memory_bank_id if primary_was_supplied else current.primary_memory_bank_id
            if req.memory_bank_ids is not None and requested_primary not in requested_memory_ids:
                requested_primary = requested_memory_ids[0] if requested_memory_ids else None
            memory_ids, primary_memory_id = _memory_binding(requested_memory_ids, requested_primary)
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
                    "model": "" if current.app_type == "workflow" else (req.model if req.model is not None else current.model),
                    "system_prompt": req.system_prompt if req.system_prompt is not None else current.system_prompt,
                    "avatar_url": req.avatar_url if req.avatar_url is not None else current.avatar_url,
                    "tool_ids": req.tool_ids if req.tool_ids is not None else current.tool_ids,
                    "skill_ids": req.skill_ids if req.skill_ids is not None else current.skill_ids,
                    "knowledge_base_ids": req.knowledge_base_ids if req.knowledge_base_ids is not None else current.knowledge_base_ids,
                    "knowledge_base_bindings": req.knowledge_base_bindings if req.knowledge_base_bindings is not None else current.knowledge_base_bindings,
                    "memory_bank_ids": memory_ids,
                    "primary_memory_bank_id": primary_memory_id,
                    "memory_config": normalize_memory_config(req.memory_config if req.memory_config is not None else current.memory_config),
                    "prompt_variables": req.prompt_variables if req.prompt_variables is not None else current.prompt_variables,
                    "metadata": req.metadata if req.metadata is not None else current.metadata,
                }
            )
            if updated.workflow_id:
                workflow = workflows.get(updated.workflow_id)
                if updated.app_type == "workflow":
                    errors = _validate_application_workflow(workflow.graph, app_tool_ids=updated.tool_ids)
                    if errors:
                        raise ValueError("；".join(errors))
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
                            "knowledge_base_bindings": updated.knowledge_base_bindings,
                            "memory_bank_ids": updated.memory_bank_ids,
                            "primary_memory_bank_id": updated.primary_memory_bank_id,
                            "memory_config": updated.memory_config,
                            "prompt_variables": updated.prompt_variables,
                        },
                    )
                workflow_update = WorkflowRecord.from_dict(
                    {
                        **workflow.to_dict(),
                        "name": updated.name,
                        "description": updated.description,
                        "graph": graph.to_dict(),
                    }
                )
                workflows.save_draft(workflow_update)
            return applications.save(updated).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/apps/{app_id}/publish")
    def publish_application(app_id: str, request: Request) -> Dict[str, Any]:
        try:
            app_record = _owned_application(app_id, request)
            workflow = workflows.get(app_record.workflow_id) if app_record.workflow_id else None
            model_errors = _model_selection_errors(app_record, workflow.graph if workflow else None)
            if model_errors:
                raise ValueError("；".join(model_errors))
            if app_record.app_type == "workflow" and workflow is not None:
                errors = _validate_application_workflow(workflow.graph, runnable=True, app_tool_ids=app_record.tool_ids)
                if errors:
                    raise ValueError("；".join(errors))
            if workflow is not None:
                published = workflows.publish(app_record.workflow_id)
                app_record.metadata = {**app_record.metadata, "workflow_version": published.version}
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
            model_errors = _model_selection_errors(app_record, workflows.get(app_record.workflow_id).graph)
            if model_errors:
                raise HTTPException(status_code=400, detail="；".join(model_errors))
            if app_record.app_type == "workflow":
                errors = _validate_application_workflow(workflows.get(app_record.workflow_id).graph, runnable=True, app_tool_ids=app_record.tool_ids)
                if errors:
                    raise HTTPException(status_code=400, detail="；".join(errors))
            input_payload = dict(req.input or {})
            workspace_id = str(app_record.metadata.get("workspace_id") or "").strip()
            if workspace_id and not input_payload.get("workspace_id"):
                input_payload["workspace_id"] = workspace_id
            memory_config = normalize_memory_config(app_record.memory_config)
            conversation = None
            if memory_config["short_term_enabled"]:
                conversation = conversations.get(req.conversation_id) if req.conversation_id else conversations.create(app_record.id, app_record.owner_user_id)
                if conversation.application_id != app_record.id or conversation.owner_user_id not in {"", app_record.owner_user_id}:
                    raise HTTPException(status_code=404, detail="会话不存在")
            try:
                input_payload["variables"] = _validated_variables(app_record.prompt_variables, input_payload.get("variables"))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            if "input" not in input_payload:
                input_payload["input"] = f"请运行应用：{app_record.name}"
            current_text = str(input_payload.get("input") or "")
            context_usage = build_conversation_context(conversation, memory_config, current_text) if conversation else {"text":"","messages":[],"summary":"","rounds":0,"estimated_tokens":0,"compressed":False}
            input_payload["__conversation_context_text__"] = context_usage.pop("text")
            if conversation:
                conversation.messages.append({"role":"user","content":current_text,"created_at":_utc_now()})
                conversation.rolling_summary = context_usage.get("summary") or conversation.rolling_summary
                conversations.save(conversation)
            record = runs.create(
                input=input_payload,
                recursion_limit=req.recursion_limit,
                workflow_id=app_record.workflow_id,
            )
            record.metadata["application_id"] = app_record.id
            record.metadata["application_name"] = app_record.name
            record.metadata["owner_user_id"] = app_record.owner_user_id
            record.metadata["conversation_id"] = conversation.id if conversation else None
            record.metadata["context_usage"] = context_usage
            record.status = "queued"
            runs.save(record)
            background_tasks.add_task(_execute_run, record)
            return record.to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/api/apps/{app_id}/conversations")
    def create_conversation(app_id: str, request: Request) -> Dict[str, Any]:
        app_record = _owned_application(app_id, request)
        return conversations.create(app_record.id, app_record.owner_user_id).to_dict()

    @app.get("/api/apps/{app_id}/conversations")
    def list_conversations(app_id: str, request: Request) -> List[Dict[str, Any]]:
        app_record = _owned_application(app_id, request)
        return [item.to_dict() for item in conversations.list(app_record.id, app_record.owner_user_id)]

    @app.get("/api/apps/{app_id}/conversations/{conversation_id}")
    def get_conversation(app_id: str, conversation_id: str, request: Request) -> Dict[str, Any]:
        app_record = _owned_application(app_id, request)
        try:
            item = conversations.get(conversation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        if item.application_id != app_record.id or item.owner_user_id not in {"", app_record.owner_user_id}:
            raise HTTPException(status_code=404, detail="会话不存在")
        return item.to_dict()

    @app.delete("/api/apps/{app_id}/conversations/{conversation_id}")
    def clear_conversation(app_id: str, conversation_id: str, request: Request) -> Dict[str, Any]:
        get_conversation(app_id, conversation_id, request)
        return conversations.clear(conversation_id).to_dict()

    @app.post("/api/apps/{app_id}/memory-preview")
    async def preview_application_memory(app_id: str, request: Request) -> Dict[str, Any]:
        app_record = _owned_application(app_id, request)
        body = await request.json()
        query = str(body.get("query") or "")
        conversation = None
        if body.get("conversation_id"):
            conversation = conversations.get(str(body["conversation_id"]))
            if conversation.application_id != app_record.id:
                raise HTTPException(status_code=404, detail="会话不存在")
        context = build_conversation_context(conversation, normalize_memory_config(app_record.memory_config), query) if conversation else {"text":"","messages":[],"summary":"","rounds":0,"estimated_tokens":0,"compressed":False}
        recalled = []
        top_k = normalize_memory_config(app_record.memory_config)["memory_top_k"]
        for bank_id in app_record.memory_bank_ids:
            for item in list_bank_memories(memory_data_root, bank_id, query=query, limit=top_k):
                recalled.append({**item.to_dict(),"bank_id":bank_id,"bank_role":"primary" if bank_id==app_record.primary_memory_bank_id else "reference"})
        return {"query":query,"context":context,"memory_banks":[{"id":bank_id,"role":"primary" if bank_id==app_record.primary_memory_bank_id else "reference"} for bank_id in app_record.memory_bank_ids],"recalled_memories":recalled[:top_k]}

    @app.get("/api/apps/{app_id}/memory-audit")
    def list_application_memory_audit(app_id: str, request: Request, limit: int = 100) -> List[Dict[str, Any]]:
        _owned_application(app_id, request)
        return memory_audit.list(app_id, limit)

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
            app_record = _owned_app_for_workflow(workflow_id, request)
            current = workflows.get(workflow_id)
            next_graph = req.graph if req.graph is not None else current.graph
            if "workflow" in current.tags:
                next_graph = _normalize_workflow_node_names(next_graph)
                errors = _validate_application_workflow(next_graph, app_tool_ids=app_record.tool_ids if app_record else None)
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
            return (workflows.save_draft(updated) if "application" in current.tags else workflows.save(updated)).to_dict()
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
            current = workflows.get(workflow_id)
            if "application" in current.tags:
                activated = workflows.activate_version(workflow_id, version)
                application = _owned_app_for_workflow(workflow_id, request)
                if application is not None:
                    application.metadata = {**application.metadata, "workflow_version": version}
                    if application.app_type == "agent":
                        entry = next((item for item in activated.graph.get("agents", []) if item.get("id") == application.entry_agent_id), None)
                        if entry:
                            config = dict(entry.get("config") or {})
                            application.name = str(entry.get("name") or application.name)
                            application.description = str(entry.get("description") or "")
                            application.model = str(entry.get("model") or "")
                            application.system_prompt = str(entry.get("sys_prompt") or "")
                            for key in ("tool_ids", "skill_ids", "knowledge_base_ids", "memory_bank_ids", "prompt_variables"):
                                setattr(application, key, [str(item) if key != "prompt_variables" else item for item in config.get(key) or []])
                            application.primary_memory_bank_id = config.get("primary_memory_bank_id")
                            application.memory_config = normalize_memory_config(config.get("memory_config"))
                    applications.save(application)
                return activated.to_dict()
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

    @app.post("/api/skills/import/file")
    async def import_skill_file(filename: str, request: Request) -> Dict[str, Any]:
        try:
            package = read_skill_file(await request.body(), filename)
            user_id = _request_user_id(request)
            existing = next((item for item in skills.list() if item.package_sha256 == package["sha256"] and item.owner_user_id == user_id), None)
            if existing is not None:
                return existing.to_dict()
            source_type = "zip" if filename.lower().endswith(".zip") else "manual"
            record = skills.create(
                name=package["name"], content=package["content"],
                description=request.headers.get("x-skill-description", "从文件导入"),
                status=SkillStatus.PUBLISHED, tags=["imported", "file"],
                metadata={"source":"file","filename":filename,"sha256":package["sha256"],"references":package["references"],"scripts_ignored":True,"validation":{"passed":True,"mode":"automatic"}},
                owner_user_id=user_id, visibility="private",
                source_type=source_type, validation_status="passed", package_sha256=package["sha256"],
            )
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
    # 市场只展示有明确来源的工具：平台内置工具在前端直接来自工具目录；
    # MCP 条目是可验证的远程服务或明确标注为“需配置”的连接模板。
    mcp_market = [
        {"slug":"local-demo","name":"本地演示 MCP","provider":"AgentForge","category":"开发测试","description":"零权限 JSON-RPC 演示服务，用于验证 MCP 发现、选择与调用链。","cover":"violet","mcp_url":"http://127.0.0.1:8000/mcp/demo","verified":True,"tools":[{"name":"preview_email","title":"邮件预览","description":"仅生成邮件预览，不发送真实邮件"},{"name":"lookup_demo","title":"演示检索","description":"返回本地演示检索结果"}]},
        {"slug":"context7","name":"Context7 文档 MCP","provider":"Context7","category":"开发工具","description":"查询公开的软件库与框架最新文档；无需平台密钥即可测试，工具调用仍需逐次批准。","cover":"mint","mcp_url":"https://mcp.context7.com/mcp","verified":True,"tools":[{"name":"resolve-library-id","title":"解析文档库","description":"把库名解析为 Context7 文档库 ID","inputSchema":{"type":"object","properties":{"libraryName":{"type":"string"}},"required":["libraryName"]}},{"name":"query-docs","title":"查询文档","description":"基于已解析的库 ID 查询文档","inputSchema":{"type":"object","properties":{"libraryId":{"type":"string"},"query":{"type":"string"}},"required":["libraryId","query"]}}]},
        {"slug":"deepwiki","name":"DeepWiki 仓库问答 MCP","provider":"DeepWiki","category":"开发工具","description":"对任意 GitHub 开源仓库进行 AI 问答与文档 Wiki 浏览；官方公开服务，无需平台密钥即可测试，工具调用仍需逐次批准。","cover":"cyan","mcp_url":"https://mcp.deepwiki.com/mcp","verified":True,"tools":[{"name":"ask_question","title":"仓库问答","description":"针对 GitHub 仓库提问并获得基于上下文的回答","inputSchema":{"type":"object","properties":{"repoName":{"type":"string","description":"owner/repo 格式的仓库名"},"question":{"type":"string"}},"required":["repoName","question"]}},{"name":"read_wiki_structure","title":"读取文档结构","description":"获取仓库文档 Wiki 的主题目录","inputSchema":{"type":"object","properties":{"repoName":{"type":"string"}},"required":["repoName"]}},{"name":"read_wiki_contents","title":"阅读文档内容","description":"阅读仓库 DeepWiki 文档内容","inputSchema":{"type":"object","properties":{"repoName":{"type":"string"}},"required":["repoName"]}}]},
        {"slug":"ms-learn","name":"Microsoft Learn 文档 MCP","provider":"Microsoft","category":"文档检索","description":"检索 Microsoft Learn 与 Azure 官方文档、代码示例并转为 Markdown；官方公开服务，无需平台密钥即可测试，工具调用仍需逐次批准。","cover":"violet","mcp_url":"https://learn.microsoft.com/api/mcp","verified":True,"tools":[{"name":"microsoft_docs_search","title":"搜索官方文档","description":"搜索 Microsoft 与 Azure 官方文档","inputSchema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}},{"name":"microsoft_code_sample_search","title":"搜索代码示例","description":"在官方文档中检索代码片段与示例","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"language":{"type":"string"}},"required":["query"]}},{"name":"microsoft_docs_fetch","title":"抓取文档页","description":"把 Microsoft Learn 文档页转换为 Markdown","inputSchema":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}]},
        {"slug":"cloudflare-docs","name":"Cloudflare 文档 MCP","provider":"Cloudflare","category":"云服务文档","description":"检索 Cloudflare 产品（Workers、R2、DNS 等）官方文档；官方公开服务，无需平台密钥即可测试，工具调用仍需逐次批准。","cover":"mint","mcp_url":"https://docs.mcp.cloudflare.com/sse","verified":True,"tools":[{"name":"search_cloudflare_documentation","title":"搜索 Cloudflare 文档","description":"检索 Cloudflare 产品官方文档并返回相关章节","inputSchema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}},{"name":"migrate_pages_to_workers_guide","title":"Pages 迁移指南","description":"获取 Pages 项目迁移到 Workers 的官方指南","inputSchema":{"type":"object","properties":{}}}]},
        {"slug":"exa","name":"Exa 联网检索 MCP","provider":"Exa","category":"联网检索","description":"语义化网页搜索与正文抓取，获取模型训练截止之后的实时信息；公开端点可直接测试，工具调用仍需逐次批准。","cover":"cyan","mcp_url":"https://mcp.exa.ai/mcp","verified":True,"tools":[{"name":"web_search_exa","title":"语义网页搜索","description":"按自然语言语义检索网页并返回干净正文","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"numResults":{"type":"number"}},"required":["query"]}},{"name":"web_fetch_exa","title":"抓取网页正文","description":"把指定 URL 的网页内容抓取为干净 Markdown","inputSchema":{"type":"object","properties":{"urls":{"type":"array","items":{"type":"string"}},"maxCharacters":{"type":"number"}},"required":["urls"]}}]},
        {"slug":"web-search","name":"联网检索 MCP 模板","provider":"项目精选目录","category":"通用办公","description":"接入组织已采购的检索 MCP；安装后需要填写实际服务地址和凭据。","cover":"mint","verified":False},
        {"slug":"gmail","name":"Gmail MCP","provider":"Google","category":"邮件办公","description":"使用 Google OAuth 创建草稿、读取或发送 Gmail；普通用户连接账号即可。当前 Google 官方 MCP 需要管理员预先配置 OAuth 应用。","cover":"cyan","mcp_url":"https://gmailmcp.googleapis.com/mcp/v1","verified":False,"connection_schema":GMAIL_STATIC_OAUTH_SCHEMA},
        {"slug":"github","name":"GitHub MCP","provider":"GitHub","category":"代码协作","description":"用于仓库、Issue 与 PR 协作；服务真实可用，但需在自定义连接中配置 GitHub 认证后再安装。","cover":"violet","verified":False},
        {"slug":"notion","name":"Notion 工作区 MCP","provider":"Notion","category":"知识库","description":"连接 Notion 工作区，搜索与更新页面、数据库；官方服务真实可用，需在自定义连接中填写 https://mcp.notion.com/mcp 并完成 OAuth 登录授权后安装。","cover":"mint","verified":False},
        {"slug":"slack","name":"Slack 协作 MCP","provider":"Slack","category":"即时通讯","description":"读写 Slack 频道消息与串、整理团队讨论；官方服务真实可用，需在自定义连接中填写 https://mcp.slack.com/mcp 并完成 OAuth 登录授权后安装。","cover":"violet","verified":False},
        {"slug":"sentry","name":"Sentry 监控 MCP","provider":"Sentry","category":"监控运维","description":"查询错误事件、Issue 堆栈与发布状态；官方服务真实可用，需在自定义连接中填写 https://mcp.sentry.dev/mcp 并完成 OAuth 登录授权后安装。","cover":"cyan","verified":False},
        {"slug":"amap","name":"高德地图 MCP","provider":"高德开放平台","category":"位置服务","description":"地理编码、POI 检索与驾车/步行路线规划；需在高德开放平台申请 Key 后，在自定义连接中填写服务地址与 Key 再安装。","cover":"mint","verified":False},
        {"slug":"tavily","name":"Tavily 联网检索 MCP","provider":"Tavily","category":"信息检索","description":"面向大模型的实时联网检索与网页内容抽取；需在 tavily.com 申请 API Key 后，在自定义连接中填写服务地址与密钥再安装。","cover":"violet","verified":False},
        {"slug":"enterprise-knowledge","name":"企业知识检索 MCP 模板","provider":"项目精选目录","category":"知识库","description":"接入企业文档检索服务；安装后需要填写实际 MCP 地址。","cover":"cyan","verified":False},
    ]
    def _tool_connection_payload(connection: ToolConnectionRecord) -> Dict[str, Any]:
        child_tools = []
        for tool_id in connection.tool_ids:
            try:
                child_tools.append(tools.get(tool_id).to_dict())
            except KeyError:
                continue
        payload = {**connection.to_dict(), "tools": child_tools, "tool_count": len(child_tools)}
        if connection.type == "mcp":
            payload["authorization_required"] = connection.status == "authorization_required"
            payload["authorized"] = bool(_connection_mcp_token(connection))
        return payload

    def _oauth_redirect_uri(request: Request) -> str:
        public_base = os.environ.get("PUBLIC_API_BASE_URL", "").strip().rstrip("/")
        return (public_base or str(request.base_url).rstrip("/")) + "/api/tool-connections/oauth/callback"

    def _connection_mcp_token(connection: ToolConnectionRecord) -> str:
        if connection.credential_env:
            return os.environ.get(connection.credential_env, "")
        return mcp_oauth.bearer_token_for(connection.id) or mcp_oauth.token_for(connection.id)

    def _mcp_schema_for_endpoint(url: str, timeout: float, schema_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if url.rstrip("/") == "https://gmailmcp.googleapis.com/mcp/v1":
            return {**GMAIL_STATIC_OAUTH_SCHEMA, "protocol_source":"market_override", "tool_count":0}
        discovered = inspect_connection(url, timeout=timeout)
        # Marketplace schemas may describe non-secret URL variables and labels,
        # but protocol-discovered authentication always wins.
        if schema_hint and discovered.get("auth_type") == "none":
            return {**schema_hint, "protocol_source":"market_schema", "tool_count":discovered.get("tool_count", 0)}
        return discovered

    def _install_discovered_mcp_tools(connection: ToolConnectionRecord, remote_tools: List[Dict[str, Any]], timeout: float) -> List[Dict[str, Any]]:
        connected = [item for item in tools.list() if item.metadata.get("connection_id") == connection.id]
        by_name = {str(item.metadata.get("remote_tool_name")): item for item in connected}
        imported: List[Dict[str, Any]] = []
        for remote in remote_tools:
            remote_name = str(remote["name"])
            annotations = dict(remote.get("annotations") or {})
            risk = "read" if annotations.get("readOnlyHint") is True else "high"
            metadata = {"source":"mcp","adapter":"mcp_http","connection_id":connection.id,"connection_name":connection.name,"mcp_url":connection.endpoint,"method":"tools/call","remote_tool_name":remote_name,"input_schema":remote.get("inputSchema") or remote.get("input_schema") or {},"credential_env":connection.credential_env,"risk":risk,"mcp_annotations":annotations,"sync_status":"synced","timeout_seconds":timeout,"auth_mode":str(connection.metadata.get("auth_mode") or ("bearer" if connection.credential_env else "oauth" if mcp_oauth.connected(connection.id) else "none"))}
            existing = by_name.get(remote_name)
            if existing:
                existing.display_name=str(remote.get("title") or remote_name); existing.description=str(remote.get("description") or existing.description); existing.metadata=metadata
                imported.append(tools.save(existing).to_dict())
            else:
                slug = f"mcp_{connection.id}_{remote_name}".replace("-", "_")
                record = tools.create(name=slug,display_name=str(remote.get("title") or remote_name),description=str(remote.get("description") or f"{connection.name} MCP 工具"),category="mcp",tags=["mcp","external"],metadata=metadata)
                connection.tool_ids.append(record.id); imported.append(record.to_dict())
        fresh_names = {str(item["name"]) for item in remote_tools}
        for stale in connected:
            if str(stale.metadata.get("remote_tool_name")) not in fresh_names:
                stale.enabled=False; stale.metadata={**stale.metadata,"sync_status":"missing"}; tools.save(stale)
        connection.tool_ids=list(dict.fromkeys(connection.tool_ids)); connection.status="installed"; connection.last_synced_at=_utc_now(); tool_connections.save(connection)
        return imported

    def _ensure_legacy_tool_connections() -> None:
        """把旧版扁平外部工具按连接信息归组，保持工具 ID 不变。"""
        known_tool_ids = {tool_id for item in tool_connections.list() for tool_id in item.tool_ids}
        groups: Dict[str, List[ToolRecord]] = {}
        for tool in tools.list():
            source = str(tool.metadata.get("source") or "")
            if source not in {"mcp", "openapi"} or tool.id in known_tool_ids:
                continue
            key = str(tool.metadata.get("connection_id") or tool.metadata.get("market_slug") or tool.metadata.get("mcp_url") or tool.metadata.get("operation_url") or tool.id)
            groups.setdefault(key, []).append(tool)
        for key, children in groups.items():
            seed = children[0]
            source = str(seed.metadata.get("source") or "mcp")
            market_slug = str(seed.metadata.get("market_slug") or "")
            connection_id = str(seed.metadata.get("connection_id") or (f"market-{market_slug}" if market_slug else f"legacy-{seed.id}"))
            if tool_connections.exists(connection_id):
                continue
            connection = tool_connections.create(id=connection_id,type=source,name=str(seed.metadata.get("connection_name") or seed.display_name),source="market" if market_slug else "custom",market_slug=market_slug,endpoint=str(seed.metadata.get("mcp_url") or seed.metadata.get("operation_url") or ""),status="configuration_required" if seed.metadata.get("needs_configuration") else "installed",credential_env=str(seed.metadata.get("credential_env") or ""),tool_ids=[item.id for item in children],last_synced_at=str(seed.updated_at))
            for child in children:
                child.metadata={**child.metadata,"connection_id":connection.id,"connection_name":connection.name};tools.save(child)

    _ensure_legacy_tool_connections()
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
        installed = {item.market_slug:item for item in tool_connections.list() if item.market_slug}
        return [{**{k:v for k,v in item.items() if k != "tools"},"tool_count":len(item.get("tools") or []),"requires_configuration":not bool(item.get("mcp_url")) or bool((item.get("connection_schema") or {}).get("fields")),"availability":"ready" if item.get("verified") and item.get("mcp_url") else "configuration_required","installed":item["slug"] in installed,"connection_id":installed[item["slug"]].id if item["slug"] in installed else ""} for item in mcp_market]

    @app.post("/api/marketplace/mcp/{slug}/install")
    def install_mcp_template(slug: str) -> Dict[str, Any]:
        item = next((x for x in mcp_market if x["slug"] == slug), None)
        if item is None:
            raise HTTPException(status_code=404, detail="未找到 MCP 市场模板")
        existing = next((x for x in tool_connections.list() if x.market_slug == slug), None)
        if existing is not None:
            payload=_tool_connection_payload(existing)
            return {"installed":False,"connection":payload,"tool":payload["tools"][0] if payload["tools"] else None,"message":"该 MCP 已安装到工作区"}
        connection = tool_connections.create(id=f"market-{slug}",type="mcp",name=item["name"],source="market",market_slug=slug,endpoint=item.get("mcp_url", ""),status="installed" if item.get("mcp_url") else "configuration_required",metadata={"provider":item.get("provider", ""),"verified":bool(item.get("verified"))})
        remote_tools = item.get("tools") or []
        # Fetch the live schema where possible.  The bundled schema remains a
        # safe fallback so the marketplace stays usable if a remote service is
        # temporarily unavailable during installation.
        if connection.endpoint.startswith("https://"):
            try:
                remote_tools = discover_mcp_tools(connection.endpoint, "", 12)
            except Exception:
                pass
        for remote in remote_tools:
            annotations = dict(remote.get("annotations") or {})
            risk = "read" if annotations.get("readOnlyHint") is True else "high"
            record = tools.create(name=f"mcp_{slug}_{remote['name']}",display_name=str(remote.get("title") or remote["name"]),description=str(remote.get("description") or "MCP 工具"),category="mcp",tags=["mcp","market"],metadata={"source":"mcp","adapter":"mcp_http","connection_id":connection.id,"market_slug":slug,"mcp_url":connection.endpoint,"method":"tools/call","remote_tool_name":remote["name"],"input_schema":remote.get("inputSchema") or remote.get("input_schema") or {"type":"object"},"risk":risk,"mcp_annotations":annotations,"sync_status":"synced","needs_configuration":False})
            connection.tool_ids.append(record.id)
        connection.last_synced_at=_utc_now() if connection.tool_ids else "";tool_connections.save(connection)
        message = "MCP 已安装到工作区，请按需添加子工具" if connection.tool_ids else "MCP 模板已安装，请先完成服务配置"
        payload=_tool_connection_payload(connection)
        return {"installed":True,"connection":payload,"tool":payload["tools"][0] if payload["tools"] else None,"message":message}

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
        result = []
        for bank in memory_banks.list():
            items = list_bank_memories(memory_data_root, bank.id, limit=500)
            bindings = [app_item for app_item in applications.list() if bank.id in app_item.memory_bank_ids]
            result.append({**bank.to_dict(), "memory_count":len(items), "scope_counts":_scope_counts(items), "binding_count":len(bindings)})
        return result

    @app.post("/api/memory-banks")
    def create_memory_bank(req: CreateMemoryBankReq) -> Dict[str, Any]:
        try:
            return memory_banks.create(name=req.name, description=req.description, metadata=req.metadata).to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.put("/api/memory-banks/{bank_id}")
    def update_memory_bank(bank_id: str, req: UpdateMemoryBankReq) -> Dict[str, Any]:
        try:
            bank=memory_banks.get(bank_id)
            if req.name is not None:
                value=req.name.strip()
                if not value or len(value)>32: raise ValueError("记忆库名称长度必须为 1–32")
                bank.name=value
            if req.description is not None:
                value=req.description.strip()
                if not value or len(value)>128: raise ValueError("记忆库描述长度必须为 1–128")
                bank.description=value
            if req.retrieval_config is not None: bank.retrieval_config=normalize_memory_retrieval_config(req.retrieval_config)
            return memory_banks.save(bank).to_dict()
        except KeyError as exc: raise HTTPException(status_code=404,detail=str(exc))
        except ValueError as exc: raise HTTPException(status_code=400,detail=str(exc))

    def _bank_rules(bank_id: str) -> tuple[Any,List[Dict[str,Any]]]:
        bank=memory_banks.get(bank_id);return bank,list(bank.metadata.get("rules") or [])

    @app.get("/api/memory-banks/{bank_id}/rules")
    def list_memory_rules(bank_id: str) -> List[Dict[str,Any]]:
        try:return _bank_rules(bank_id)[1]
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))

    @app.post("/api/memory-banks/{bank_id}/rules")
    def create_memory_rule(bank_id: str, req: MemoryRuleReq) -> Dict[str,Any]:
        try:
            bank,rules=_bank_rules(bank_id)
            if req.type not in {"fragment","profile"}:raise ValueError("规则类型必须为 fragment 或 profile")
            if sum(1 for item in rules if item.get("type")==req.type)>=50:raise ValueError("同类记忆规则最多 50 条")
            item={"id":f"rule-{os.urandom(6).hex()}",**req.model_dump(),"target_scope":"project"}
            rules.append(item);bank.metadata={**bank.metadata,"rules":rules};memory_banks.save(bank);return item
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
        except ValueError as exc:raise HTTPException(status_code=400,detail=str(exc))

    @app.put("/api/memory-banks/{bank_id}/rules/{rule_id}")
    def update_memory_rule(bank_id: str, rule_id: str, req: MemoryRuleReq) -> Dict[str,Any]:
        try:
            bank,rules=_bank_rules(bank_id);index=next((i for i,x in enumerate(rules) if x.get("id")==rule_id),-1)
            if index<0:raise KeyError("memory rule not found")
            rules[index]={"id":rule_id,**req.model_dump(),"target_scope":"project"};bank.metadata={**bank.metadata,"rules":rules};memory_banks.save(bank);return rules[index]
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))

    @app.post("/api/memory-banks/{bank_id}/rules/{rule_id}/copy")
    def copy_memory_rule(bank_id: str, rule_id: str) -> Dict[str,Any]:
        try:
            bank,rules=_bank_rules(bank_id);source=next(x for x in rules if x.get("id")==rule_id)
            if sum(1 for x in rules if x.get("type")==source.get("type"))>=50:raise ValueError("同类记忆规则最多 50 条")
            item={**source,"id":f"rule-{os.urandom(6).hex()}","name":f"{source.get('name')} 副本"};rules.append(item);bank.metadata={**bank.metadata,"rules":rules};memory_banks.save(bank);return item
        except StopIteration:raise HTTPException(status_code=404,detail="memory rule not found")
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
        except ValueError as exc:raise HTTPException(status_code=400,detail=str(exc))

    @app.delete("/api/memory-banks/{bank_id}/rules/{rule_id}")
    def delete_memory_rule(bank_id: str, rule_id: str) -> Dict[str,Any]:
        try:
            bank,rules=_bank_rules(bank_id);filtered=[x for x in rules if x.get("id")!=rule_id]
            if len(filtered)==len(rules):raise KeyError("memory rule not found")
            bank.metadata={**bank.metadata,"rules":filtered};memory_banks.save(bank);return {"ok":True}
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))

    @app.post("/api/memory-banks/{bank_id}/search")
    async def test_memory_search(bank_id: str, request: Request) -> Dict[str,Any]:
        try:
            bank=memory_banks.get(bank_id);body=await request.json();query=str(body.get("query") or "").strip();config=normalize_memory_retrieval_config({**bank.retrieval_config,**dict(body.get("config") or {})})
            items=[]
            for scope_name in config["scopes"]:
                items.extend(list_bank_memories(memory_data_root,bank_id,query=query,scope=MemoryScope(scope_name),limit=config["top_k"]))
            items=sorted({item.id:item for item in items}.values(),key=lambda item:(item.importance,item.ts),reverse=True)[:config["top_k"]]
            return {"query":query,"config":config,"items":[{**item.to_dict(),"bank_id":bank_id,"bank_role":"bank","score":item.importance} for item in items]}
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))

    @app.get("/api/memory-banks/{bank_id}/audit")
    def list_memory_bank_audit(bank_id: str, action: str = "", rule_id: str = "", limit: int = 200) -> List[Dict[str,Any]]:
        try:memory_banks.get(bank_id)
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
        rows=[]
        for app_item in applications.list():
            rows.extend(item for item in memory_audit.list(app_item.id,limit) if item.get("bank_id")==bank_id)
        if action:rows=[item for item in rows if item.get("action")==action]
        if rule_id:rows=[item for item in rows if item.get("rule_id")==rule_id]
        return sorted(rows,key=lambda item:str(item.get("timestamp") or ""),reverse=True)[:max(1,min(limit,500))]

    @app.get("/api/memory-banks/{bank_id}")
    def get_memory_bank(bank_id: str) -> Dict[str, Any]:
        try:
            bank = memory_banks.get(bank_id)
            items = list_bank_memories(memory_data_root, bank_id, limit=500)
            bindings = [{"id":item.id,"name":item.name,"primary":item.primary_memory_bank_id==bank_id} for item in applications.list() if bank_id in item.memory_bank_ids]
            return {**bank.to_dict(),"memory_count":len(items),"scope_counts":_scope_counts(items),"bindings":bindings}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/memory-banks/{bank_id}/memories")
    def list_memories(bank_id: str, query: str = "", scope: Optional[str] = None, offset: int = 0, limit: int = 100) -> Dict[str, Any]:
        try:
            memory_banks.get(bank_id)
            effective_scope = MemoryScope(scope) if scope else None
            items = list_bank_memories(memory_data_root, bank_id, query=query, scope=effective_scope, limit=500)
            bounded_limit = max(1, min(limit, 200))
            return {"items":[item.to_dict() for item in items[max(0,offset):max(0,offset)+bounded_limit]],"total":len(items)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/model-connections/{connection_id}")
    def delete_model_connection(connection_id: str) -> Dict[str, Any]:
        try:
            current = model_connections.get(connection_id)
            if current.auto_default:
                raise ValueError("请先替换或取消该连接的 AUTO 默认配置，再删除")
            references: List[str] = []
            for application in applications.list():
                if application.model == connection_id:
                    references.append(f"应用“{application.name}”")
                if not application.workflow_id or not workflows.exists(application.workflow_id):
                    continue
                workflow = workflows.get(application.workflow_id)
                for node in workflow.graph.get("agents", []):
                    if str(node.get("model") or "") == connection_id:
                        references.append(f"工作流“{application.name}”的节点“{node.get('name')}”")
            if references:
                raise ValueError(f"该模型正在被{'、'.join(dict.fromkeys(references))}使用，请先解除绑定")
            model_connections.delete(connection_id)
            return {"ok": True, "id": connection_id}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/memory-banks/{bank_id}/memories")
    def create_memory(bank_id: str, req: CreateMemoryReq) -> Dict[str, Any]:
        try:
            memory_banks.get(bank_id)
            scope = MemoryScope(req.scope)
            context = MemoryContext(**{f"{scope.value}_id": req.scope_id}) if req.scope_id else None
            store = HybridTieredMemoryStore(os.path.join(memory_data_root, bank_id))
            try:
                item = store.append(req.content, scope, context=context, tags=req.tags, **{**req.metadata,"source":"manual","memory_bank_id":bank_id})
                return item.to_dict()
            finally:
                store.close()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/memory-banks/{bank_id}/memories/{memory_id}")
    def delete_memory(bank_id: str, memory_id: str) -> Dict[str, Any]:
        try:
            memory_banks.get(bank_id)
            store = HybridTieredMemoryStore(os.path.join(memory_data_root, bank_id))
            try: store.delete(memory_id)
            finally: store.close()
            return {"ok":True}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/memory-banks/{bank_id}/memories/{memory_id}/promote")
    def promote_memory(bank_id: str, memory_id: str) -> Dict[str,Any]:
        try:
            memory_banks.get(bank_id);source=next((item for item in list_bank_memories(memory_data_root,bank_id,limit=500) if item.id==memory_id),None)
            if source is None:raise KeyError("memory not found")
            store=HybridTieredMemoryStore(os.path.join(memory_data_root,bank_id))
            try:
                promoted=store.append(source.content,MemoryScope.GLOBAL,tags=list(source.tags)+["promoted"],**{**source.metadata,"promoted_from":source.id,"source":"manual_promotion"});store.delete(source.id)
            finally:store.close()
            return promoted.to_dict()
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))

    @app.post("/api/apps/{app_id}/memory-audit/{audit_id}/undo")
    def undo_memory_audit(app_id: str, audit_id: str, request: Request) -> Dict[str,Any]:
        app_record=_owned_application(app_id,request);event=next((item for item in memory_audit.list(app_id,500) if item.get("id")==audit_id),None)
        if not event or event.get("action")!="added" or not event.get("memory_id"):raise HTTPException(status_code=400,detail="该审计记录不可撤销")
        store=HybridTieredMemoryStore(os.path.join(memory_data_root,str(event.get("bank_id") or app_record.primary_memory_bank_id)))
        try:store.delete(str(event["memory_id"]))
        finally:store.close()
        return memory_audit.append(app_id,{"action":"undo","reason":"manual_undo","memory_id":event["memory_id"],"undo_of":audit_id,"bank_id":event.get("bank_id")})

    @app.delete("/api/memory-banks/{bank_id}/memories")
    def clear_memories(bank_id: str, confirm: bool = False) -> Dict[str, Any]:
        if not confirm:
            raise HTTPException(status_code=400, detail="清空记忆库必须显式传入 confirm=true")
        try:
            memory_banks.get(bank_id)
            store = HybridTieredMemoryStore(os.path.join(memory_data_root, bank_id))
            try: store.clear()
            finally: store.close()
            return {"ok":True}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.delete("/api/memory-banks/{bank_id}")
    def delete_memory_bank(bank_id: str) -> Dict[str, Any]:
        try:
            bindings = [{"id":item.id,"name":item.name} for item in applications.list() if bank_id in item.memory_bank_ids]
            if bindings:
                raise HTTPException(status_code=409, detail={"message":"记忆库仍被应用绑定","applications":bindings})
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
    def list_console_resources(kind: str, request: Request) -> List[Dict[str, Any]]:
        if kind not in resource_kinds:
            raise HTTPException(status_code=404, detail="不支持的资源类型")
        # Compatibility bridge for older agent/workflow editors. New code uses
        # /api/knowledge-bases; never recreate the old generic-resource shell.
        if kind == "knowledge-bases":
            return [item.to_dict() for item in knowledge.list_bases(_request_user_id(request))]
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
    @app.post("/api/tool-connections/mcp/probe")
    def probe_mcp_connection(req: Dict[str, Any], request: Request) -> Dict[str, Any]:
        """Inspect a URL using MCP/OAuth metadata; never ask an LLM."""
        try:
            url = validate_remote_url(str(req.get("url") or ""))
            market_slug = str(req.get("market_slug") or "")
            market_item = next((item for item in mcp_market if item.get("slug") == market_slug), None)
            schema_hint = dict(market_item.get("connection_schema") or {}) if market_item else None
            schema = _mcp_schema_for_endpoint(url, min(30, max(1, int(req.get("timeout_seconds") or 12))), schema_hint)
            return {"url":url, "name":str(market_item.get("name") if market_item else ""), "schema":schema, "oauth_callback_url":_oauth_redirect_uri(request) if str(schema.get("auth_type")).startswith("oauth") else ""}
        except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/tool-connections/mcp")
    def import_mcp_connection(req: Dict[str, Any], request: Request) -> Dict[str, Any]:
        try:
            url = validate_remote_url(str(req.get("url") or ""))
            name = str(req.get("name") or "Remote MCP").strip()
            credential_env = str(req.get("credential_env") or "")
            timeout = min(30, max(1, int(req.get("timeout_seconds") or 8)))
            market_slug = str(req.get("market_slug") or "")
            market_item = next((item for item in mcp_market if item.get("slug") == market_slug), None)
            schema_hint = dict(market_item.get("connection_schema") or {}) if market_item else None
            schema = _mcp_schema_for_endpoint(url, timeout, schema_hint)
            configuration = {str(key): value for key, value in dict(req.get("configuration") or {}).items()}
            required = missing_required(schema, configuration)
            if required:
                raise ValueError(f"请填写：{'、'.join(required)}")
            connection_id = f"mcp-{time.time_ns()}"
            auth_type = str(schema.get("auth_type") or "none")
            connection = tool_connections.create(id=connection_id,type="mcp",name=name,source="market" if market_item else "custom",market_slug=market_slug if market_item else "",endpoint=url,status="syncing",credential_env=credential_env,metadata={"auth_mode":auth_type,"connection_schema":schema})
            if configuration:
                mcp_oauth.save_configuration(connection.id, configuration)
            if auth_type == "manual_bearer":
                imported = _install_discovered_mcp_tools(connection, discover_mcp_tools(url, "", timeout, access_token=_connection_mcp_token(connection)), timeout)
            elif auth_type in {"oauth_dcr", "oauth_static"}:
                config = mcp_oauth.configuration(connection.id)
                authorization = start_authorization(endpoint=url, redirect_uri=_oauth_redirect_uri(request), connection_id=connection.id, store=mcp_oauth, timeout=timeout, client_id=config.get("client_id", ""), client_secret=config.get("client_secret", ""))
                connection.status="authorization_required"; connection.metadata={**connection.metadata,"auth_mode":auth_type,"authorization_started_at":_utc_now()}; tool_connections.save(connection)
                return {"connection_id":connection_id,"connection":_tool_connection_payload(connection),"tools":[],"schema":schema,"authorization":authorization,"message":"该 MCP 需要浏览器登录。请完成授权后回到这里同步工具。"}
            else:
                imported = _install_discovered_mcp_tools(connection, discover_mcp_tools(url, credential_env, timeout), timeout)
            return {"connection_id":connection_id,"connection":_tool_connection_payload(connection),"tools":imported}
        except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            if 'connection' in locals() and tool_connections.exists(connection.id): tool_connections.delete(connection.id)
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/tool-connections/oauth/callback")
    def complete_mcp_oauth(code: str = "", state: str = "", error: str = "", error_description: str = "") -> Response:
        if error:
            return Response(f"<h2>授权未完成</h2><p>{error_description or error}</p><p>请关闭此页并在产品中重新连接。</p>", status_code=400, media_type="text/html")
        try:
            connection_id = complete_authorization(state=state, code=code, store=mcp_oauth)
            connection = tool_connections.get(connection_id)
            remote_tools = discover_mcp_tools(connection.endpoint, "", 15, access_token=_connection_mcp_token(connection))
            _install_discovered_mcp_tools(connection, remote_tools, 15)
            return Response("<h2>授权成功</h2><p>工具已同步到工作区。可以关闭此窗口并回到产品继续操作。</p><script>window.opener&&window.opener.postMessage({type:'mcp-oauth-complete'},'*');</script>", media_type="text/html")
        except Exception as exc:
            return Response(f"<h2>授权后同步失败</h2><p>{str(exc)}</p><p>请关闭此页，在‘已安装’中重试同步。</p>", status_code=400, media_type="text/html")

    @app.post("/api/tool-connections/{connection_id}/authorize")
    def authorize_mcp_connection(connection_id: str, request: Request) -> Dict[str, Any]:
        try:
            connection = tool_connections.get(connection_id)
            if connection.type != "mcp":
                raise ValueError("只有 MCP 连接支持此授权流程")
            if connection.credential_env:
                raise ValueError("当前连接使用 Bearer 环境变量，无需浏览器 OAuth 登录")
            config = mcp_oauth.configuration(connection.id)
            authorization = start_authorization(endpoint=connection.endpoint, redirect_uri=_oauth_redirect_uri(request), connection_id=connection.id, store=mcp_oauth, client_id=config.get("client_id", ""), client_secret=config.get("client_secret", ""))
            connection.status="authorization_required"; connection.metadata={**connection.metadata,"auth_mode":"oauth","authorization_started_at":_utc_now()}; tool_connections.save(connection)
            return {"connection":_tool_connection_payload(connection),"authorization":authorization}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/tool-connections/openapi")
    def import_openapi_connection(req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            source_url = str(req.get("url") or "")
            content = read_remote_document(source_url) if source_url else json.dumps(req.get("document") or {}).encode("utf-8")
            document = parse_openapi(content)
            connection_id=f"openapi-{time.time_ns()}";connection=tool_connections.create(id=connection_id,type="openapi",name=str(req.get("name") or document.get("info",{}).get("title") or "OpenAPI 服务"),source="custom",endpoint=source_url,status="syncing",credential_env=str(req.get("credential_env") or ""));imported = []
            for operation in openapi_operations(document, source_url or "https://configured.invalid/openapi.json", str(req.get("credential_env") or "")):
                operation["metadata"]={**operation["metadata"],"connection_id":connection_id,"connection_name":connection.name}
                existing = next((item for item in tools.list() if item.name == operation["name"]), None)
                if existing and existing.metadata.get("connection_id") == connection_id:
                    existing.display_name=operation["display_name"];existing.description=operation["description"];existing.metadata={**existing.metadata,**operation["metadata"]}; imported.append(tools.save(existing).to_dict())
                else:
                    imported.append(tools.create(name=operation["name"],display_name=operation["display_name"],description=operation["description"],category="openapi",tags=["openapi","external"],metadata=operation["metadata"]).to_dict())
            connection.tool_ids=[item["id"] for item in imported];connection.status="installed";connection.last_synced_at=_utc_now();tool_connections.save(connection)
            return {"connection_id":connection_id,"connection":_tool_connection_payload(connection),"tools":imported}
        except (ValueError, OSError, urllib.error.URLError) as exc:
            if 'connection' in locals() and tool_connections.exists(connection.id): tool_connections.delete(connection.id)
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/tool-connections")
    def list_tool_connections() -> List[Dict[str, Any]]:
        _ensure_legacy_tool_connections()
        return [_tool_connection_payload(item) for item in tool_connections.list()]

    @app.get("/api/tool-connections/{connection_id}")
    def get_tool_connection(connection_id: str) -> Dict[str, Any]:
        try: return _tool_connection_payload(tool_connections.get(connection_id))
        except KeyError as exc: raise HTTPException(status_code=404,detail=str(exc))

    @app.post("/api/tool-connections/{connection_id}/test")
    def test_tool_catalog_connection(connection_id: str) -> Dict[str, Any]:
        try:
            connection=tool_connections.get(connection_id)
            if not connection.endpoint: raise ValueError("该连接尚未配置服务地址")
            if connection.type=="mcp": count=len(discover_mcp_tools(connection.endpoint,connection.credential_env,8,access_token=_connection_mcp_token(connection)))
            else: count=len(openapi_operations(parse_openapi(read_remote_document(connection.endpoint)),connection.endpoint,connection.credential_env))
            return {"ok":True,"connection_id":connection.id,"tool_count":count,"message":f"连接成功，发现 {count} 个工具"}
        except KeyError as exc: raise HTTPException(status_code=404,detail=str(exc))
        except (ValueError,OSError,urllib.error.URLError,json.JSONDecodeError) as exc: raise HTTPException(status_code=400,detail=str(exc))

    @app.post("/api/tool-connections/{connection_id}/sync")
    def sync_tool_connection(connection_id: str) -> Dict[str, Any]:
        try:
            connection=tool_connections.get(connection_id)
            if connection.type != "mcp": raise ValueError("当前仅支持同步 MCP 连接")
            remote_tools = discover_mcp_tools(connection.endpoint, connection.credential_env, 15, access_token=_connection_mcp_token(connection))
            synced = _install_discovered_mcp_tools(connection, remote_tools, 15)
            return {"connection_id":connection_id,"connection":_tool_connection_payload(connection),"tools":synced,"status":"synced"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/tool-connections/{connection_id}")
    def delete_tool_connection(connection_id: str) -> Dict[str, Any]:
        try:
            connection=tool_connections.get(connection_id)
            bound=[item.name for item in applications.list() if any(tool_id in item.tool_ids for tool_id in connection.tool_ids)]
            if bound: raise HTTPException(status_code=409,detail={"message":"连接仍被智能体应用使用","applications":bound})
            for tool_id in connection.tool_ids:
                if tools.exists(tool_id): tools.delete(tool_id)
            tool_connections.delete(connection_id)
            mcp_oauth.delete(connection_id)
            return {"ok":True}
        except KeyError as exc: raise HTTPException(status_code=404,detail=str(exc))

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

    @app.get("/api/workspaces")
    def list_workspaces() -> List[Dict[str, Any]]:
        return [item.to_dict() for item in workspaces.list()]

    @app.post("/api/workspaces/pick-directory")
    def pick_workspace_directory() -> Dict[str, Any]:
        try:
            selected = _choose_local_directory()
            if not selected:
                return {"canceled": True, "path": "", "name": ""}
            path = Path(selected).resolve()
            if not path.is_dir():
                raise ValueError("选择的路径不是可用文件夹")
            return {"canceled": False, "path": str(path), "name": path.name or str(path)}
        except (RuntimeError, ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/workspaces")
    def create_workspace(req: CreateWorkspaceReq) -> Dict[str, Any]:
        try:
            return workspaces.create(name=req.name, root_path=req.root_path, read_only=req.read_only,
                                     allowed_commands=req.allowed_commands or None, create_if_missing=req.create_if_missing).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/workspaces/{workspace_id}")
    def delete_workspace(workspace_id: str) -> Dict[str, Any]:
        workspaces.delete(workspace_id)
        return {"ok": True}

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
        return ToolRuntime(tools, mcp_oauth, workspaces).execute(tool, req.task, arguments=req.arguments or None).to_dict()

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

    # ------------------------- 企业知识库 ------------------------- #
    def _owned_kb(kb_id: str, request: Request):
        try:
            return knowledge.get_base(kb_id, _request_user_id(request))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/knowledge-bases")
    def list_knowledge_bases(request: Request) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in knowledge.list_bases(_request_user_id(request))]

    @app.post("/api/knowledge-bases")
    def create_knowledge_base(req: KnowledgeBaseReq, request: Request) -> Dict[str, Any]:
        try:return knowledge.create_base(req.model_dump(), _request_user_id(request)).to_dict()
        except ValueError as exc:raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/knowledge-bases/{kb_id}")
    def get_knowledge_base(kb_id: str, request: Request) -> Dict[str, Any]: return _owned_kb(kb_id,request).to_dict()

    @app.put("/api/knowledge-bases/{kb_id}")
    def update_knowledge_base(kb_id: str, req: KnowledgeBaseUpdateReq, request: Request) -> Dict[str, Any]:
        try:return knowledge.update_base(kb_id,req.model_dump(exclude_unset=True,exclude_none=True),_request_user_id(request)).to_dict()
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
        except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc))

    @app.delete("/api/knowledge-bases/{kb_id}")
    def delete_knowledge_base(kb_id: str, request: Request) -> Dict[str, Any]:
        _owned_kb(kb_id,request)
        references=[app.id for app in applications.list() if kb_id in app.knowledge_base_ids]
        if references:raise HTTPException(status_code=409,detail="知识库仍被应用引用，请先在应用中明确解除挂载："+"、".join(references))
        knowledge.delete_base(kb_id,_request_user_id(request));return {"ok":True}

    @app.post("/api/knowledge-bases/{kb_id}/documents")
    async def upload_knowledge_document(kb_id: str, request: Request, file: UploadFile = File(...), labels: str = Form("")) -> Dict[str, Any]:
        _owned_kb(kb_id,request)
        try:
            parsed_labels=json.loads(labels) if labels else []
            return knowledge.add_document(kb_id,_request_user_id(request),file.filename or "",await file.read(),parsed_labels).to_dict()
        except FileExistsError as exc:raise HTTPException(status_code=409,detail=str(exc))
        except (ValueError,json.JSONDecodeError) as exc:raise HTTPException(status_code=422,detail=str(exc))

    @app.get("/api/knowledge-bases/{kb_id}/documents")
    def list_knowledge_documents(kb_id:str,request:Request)->List[Dict[str,Any]]:
        _owned_kb(kb_id,request);return [d.to_dict() for d in knowledge.list_documents(kb_id)]
    @app.get("/api/knowledge-bases/{kb_id}/documents/{document_id}")
    def get_knowledge_document(kb_id:str,document_id:str,request:Request)->Dict[str,Any]:
        _owned_kb(kb_id,request)
        try:return knowledge.get_document(kb_id,document_id).to_dict()
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
    @app.delete("/api/knowledge-bases/{kb_id}/documents/{document_id}")
    def delete_knowledge_document(kb_id:str,document_id:str,request:Request)->Dict[str,Any]:
        try:knowledge.delete_document(kb_id,document_id,_request_user_id(request));return {"ok":True}
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
    @app.post("/api/knowledge-bases/{kb_id}/documents/{document_id}/reparse")
    def reparse_knowledge_document(kb_id:str,document_id:str,request:Request)->Dict[str,Any]:
        try:return knowledge.reparse(kb_id,document_id,_request_user_id(request)).to_dict()
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
    @app.post("/api/knowledge-bases/{kb_id}/documents/{document_id}/reindex")
    def reindex_knowledge_document(kb_id:str,document_id:str,request:Request)->Dict[str,Any]:
        try:return knowledge.reindex(kb_id,document_id,_request_user_id(request)).to_dict()
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
    @app.get("/api/knowledge-bases/{kb_id}/chunks")
    def list_knowledge_chunks(kb_id:str,request:Request,document_id:Optional[str]=None)->List[Dict[str,Any]]:
        _owned_kb(kb_id,request);return [c.to_dict() for c in knowledge.list_chunks(kb_id,document_id)]
    @app.post("/api/knowledge-bases/{kb_id}/chunks")
    async def create_knowledge_chunk(kb_id:str,request:Request)->Dict[str,Any]:
        try:return knowledge.save_chunk(kb_id,await request.json(),_request_user_id(request)).to_dict()
        except (KeyError,ValueError) as exc:raise HTTPException(status_code=422,detail=str(exc))
    @app.put("/api/knowledge-bases/{kb_id}/chunks/{chunk_id}")
    async def update_knowledge_chunk(kb_id:str,chunk_id:str,request:Request)->Dict[str,Any]:
        try:return knowledge.save_chunk(kb_id,await request.json(),_request_user_id(request),chunk_id).to_dict()
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
        except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc))
    @app.delete("/api/knowledge-bases/{kb_id}/chunks/{chunk_id}")
    def delete_knowledge_chunk(kb_id:str,chunk_id:str,request:Request)->Dict[str,Any]:knowledge.delete_chunk(kb_id,chunk_id,_request_user_id(request));return {"ok":True}
    def _retrieve(req:KnowledgeRetrieveReq,request:Request):
        try:return knowledge.retrieve(req.knowledge_base_ids,req.query,owner=_request_user_id(request),mode=req.mode,top_k=req.top_k,threshold=req.threshold,labels=req.labels,document_ids=req.document_ids,bindings=req.bindings)
        except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
        except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc))
    @app.post("/api/knowledge-retrieval")
    def knowledge_retrieval(req:KnowledgeRetrieveReq,request:Request)->Dict[str,Any]:return _retrieve(req,request)
    @app.post("/api/knowledge-bases/{kb_id}/retrieval-test")
    def knowledge_retrieval_test(kb_id:str,req:KnowledgeRetrieveReq,request:Request)->Dict[str,Any]:req.knowledge_base_ids=[kb_id];return _retrieve(req,request)
    @app.get("/api/knowledge-bases/{kb_id}/logs")
    def knowledge_logs(kb_id:str,request:Request)->List[Dict[str,Any]]:_owned_kb(kb_id,request);return knowledge.logs(kb_id)
    @app.get("/api/knowledge-bases/{kb_id}/statistics")
    def knowledge_statistics(kb_id:str,request:Request)->Dict[str,Any]:_owned_kb(kb_id,request);return knowledge.statistics(kb_id)

    return app


def _load_context_policy(context_policy_path: Optional[str]) -> Optional[ContextPolicy]:
    _load_dotenv_for_context_policy()
    path = context_policy_path or os.environ.get("CONTEXT_POLICY_PATH")
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_absolute() and not resolved.exists():
        resolved = Path(__file__).resolve().parents[2] / resolved
    return ContextPolicy.from_file(resolved)


# 便于 `uvicorn engine.server.app:app` 直接启动。
_load_dotenv_for_context_policy()
_default_context_policy_path = os.environ.get("CONTEXT_POLICY_PATH")
if not _default_context_policy_path and os.path.exists("configs/context_policy.yaml"):
    _default_context_policy_path = "configs/context_policy.yaml"
app = create_app(
    context_policy_path=_default_context_policy_path,
    auth_required=os.environ.get("AUTH_REQUIRED", "1") == "1",
)
