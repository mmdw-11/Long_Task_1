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

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - 取决于运行环境
    raise ImportError(
        "启动 REST 服务需要安装 fastapi 与 uvicorn：pip install fastapi 'uvicorn[standard]'"
    ) from exc

from ..constants import END
from ..modules.agent_runtime import AgentRuntimeFactory
from ..modules.context import ContextPolicy
from ..modules.product_ops import ProductStatusService, ToolCatalogStore, ToolRecord
from ..modules.security_ops import ApiAuditRecord, ApiAuditStore, utc_now
from ..modules.skills import (
    SkillEvolutionService,
    SkillRepository,
    SkillRetriever,
    SkillStatus,
    SkillTraceStore,
)
from ..modules.workflows import RunRecord, RunStore, WorkflowRecord, WorkflowStore
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


def _resolve_target(target: str) -> Any:
    """把接口传入的字符串目标解析为内部值（"END" -> END 哨兵）。"""
    return END if target == "END" else target


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    api_audit_store: Optional[ApiAuditStore] = None,
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
    tools = tool_catalog_store or ToolCatalogStore(
        os.environ.get("TOOL_CATALOG_ROOT") or "runs/tool_catalog"
    )
    api_audit = api_audit_store or ApiAuditStore(
        os.environ.get("API_AUDIT_LOG_PATH") or "runs/audit/api_audit.jsonl"
    )
    skill_traces = skill_trace_store or SkillTraceStore(
        os.environ.get("SKILL_TRACE_ROOT") or "runs/skill_traces"
    )
    skill_retriever = SkillRetriever(skills)
    skill_evolution = SkillEvolutionService(
        repository=skills,
        run_store=runs,
        trace_store=skill_traces,
    )
    runtime_factory = node_factory or AgentRuntimeFactory()
    admin_api_key = os.environ.get("ADMIN_API_KEY", "").strip()
    system_status = ProductStatusService(
        workflow_root=str(workflows.root_dir),
        run_root=str(runs.root_dir),
        skill_root=str(skills.root_dir),
        tool_root=str(tools.root_dir),
    )
    if policy is not None:
        orch.set_context_policy(policy, ledger_root=ledger_root)
    orch.set_skill_retriever(skill_retriever)
    orch.set_skill_trace_store(skill_traces)
    app = FastAPI(title="Agent 编排服务", version="0.1.0")

    @app.middleware("http")
    async def admin_guard_and_audit(request: Request, call_next):
        # 仅对写接口启用最小保护；读接口保留无密钥可访问能力。
        requires_admin = (
            request.method.upper() in {"POST", "PUT", "DELETE"}
            and request.url.path.startswith("/api/")
        )
        actor = request.headers.get("X-Actor", "")
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

    async def _execute_run(record: RunRecord) -> None:
        try:
            record.status = "running"
            runs.save(record)
            target = _orchestrator_for_run(record.workflow_id)
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
                # 长任务取消采用协作式检查，避免强杀执行线程导致状态文件损坏。
                latest = runs.get(record.id)
                if latest.status == "cancel_requested":
                    record.status = "canceled"
                    record.canceled_at = latest.canceled_at or _utc_now()
                    record.finished_at = record.canceled_at
                    record.metadata = latest.metadata
                    runs.save(record)
                    return
                record.events.append(event)
                if event.get("type") == "final":
                    record.state = dict(event.get("state") or {})
                runs.save(record)
            record.status = "succeeded"
            record.finished_at = _utc_now()
            runs.save(record)
        except Exception as e:  # noqa: BLE001 - API persists failures for polling
            record.status = "failed"
            record.error = str(e)
            record.finished_at = _utc_now()
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

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str, req: CancelRunReq) -> Dict[str, Any]:
        try:
            return runs.mark_cancel_requested(run_id, reason=req.reason).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

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
            "workflows": {"total": len(workflows.list())},
            "skills": {"total": len(skills.list())},
            "tools": {"total": len(tools.list())},
        }

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
app = create_app()
