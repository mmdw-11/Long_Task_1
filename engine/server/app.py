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
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - 取决于运行环境
    raise ImportError(
        "启动 REST 服务需要安装 fastapi 与 uvicorn：pip install fastapi 'uvicorn[standard]'"
    ) from exc

from ..constants import END
from ..modules.context import ContextPolicy
from ..modules.workflows import WorkflowRecord, WorkflowStore
from ..orchestrator import Orchestrator, _load_dotenv_for_context_policy


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


def _resolve_target(target: str) -> Any:
    """把接口传入的字符串目标解析为内部值（"END" -> END 哨兵）。"""
    return END if target == "END" else target


def create_app(
    orchestrator: Optional[Orchestrator] = None,
    *,
    context_policy: Optional[ContextPolicy] = None,
    context_policy_path: Optional[str] = None,
    context_ledger_root: Optional[str] = None,
    workflow_store: Optional[WorkflowStore] = None,
) -> "FastAPI":
    """创建并返回 FastAPI 应用。可注入已有 Orchestrator，便于测试。"""
    orch = orchestrator or Orchestrator()
    policy = context_policy or _load_context_policy(context_policy_path)
    ledger_root = context_ledger_root or os.environ.get("CONTEXT_LEDGER_ROOT") or "runs/context"
    workflows = workflow_store or WorkflowStore(
        os.environ.get("WORKFLOW_STORE_ROOT") or "runs/workflows"
    )
    if policy is not None:
        orch.set_context_policy(policy, ledger_root=ledger_root)
    app = FastAPI(title="Agent 编排服务", version="0.1.0")

    def _apply_policy(target: Orchestrator) -> None:
        if policy is not None:
            target.set_context_policy(policy, ledger_root=ledger_root)

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
            compiled = orch.build_graph(recursion_limit=req.recursion_limit)
            state = await compiled.ainvoke(req.input, req.recursion_limit)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"state": state}

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

    @app.delete("/api/workflows/{workflow_id}")
    def delete_workflow(workflow_id: str) -> Dict[str, Any]:
        try:
            workflows.delete(workflow_id)
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
