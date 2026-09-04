"""Safe, application-owned tools for local software engineering workspaces.

The module deliberately does not expose an unrestricted shell.  A workspace is
explicitly registered first, every path is resolved beneath that workspace, and
write/command operations are intended to pass through the existing approval
gate in :mod:`tool_runtime`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WorkspaceRecord:
    id: str
    name: str
    root_path: str
    read_only: bool = False
    allowed_commands: list[str] = field(default_factory=lambda: ["pytest", "npm_test", "npm_run_build", "npm_run_lint", "python_compile", "javac_compile"])
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "root_path": self.root_path, "read_only": self.read_only,
                "allowed_commands": list(self.allowed_commands), "created_at": self.created_at, "updated_at": self.updated_at}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceRecord":
        name, root = str(data.get("name") or "").strip(), str(data.get("root_path") or "").strip()
        if not name or not root:
            raise ValueError("workspace name and root_path are required")
        allowed_commands = [str(x) for x in data.get("allowed_commands") or []] or cls("", "x", "x").allowed_commands
        if "python_compile" in allowed_commands and "javac_compile" not in allowed_commands:
            allowed_commands.append("javac_compile")
        return cls(id=str(data.get("id") or f"workspace-{uuid.uuid4().hex[:12]}"), name=name, root_path=root,
                   read_only=bool(data.get("read_only", False)), allowed_commands=allowed_commands,
                   created_at=str(data.get("created_at") or _now()), updated_at=str(data.get("updated_at") or _now()))


class WorkspaceStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir or os.environ.get("WORKSPACE_STORE_ROOT") or "runs/workspaces")
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create(self, *, name: str, root_path: str, read_only: bool = False, allowed_commands: list[str] | None = None, create_if_missing: bool = False) -> WorkspaceRecord:
        root = Path(root_path).expanduser().resolve()
        if root == Path(root.anchor):
            raise ValueError("不能把磁盘根目录注册为工作区")
        if not root.exists() and create_if_missing:
            root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise ValueError("工作区目录不存在或不是目录")
        record = WorkspaceRecord(id=f"workspace-{uuid.uuid4().hex[:12]}", name=name.strip(), root_path=str(root), read_only=read_only,
                                 allowed_commands=allowed_commands or WorkspaceRecord("", "x", "x").allowed_commands)
        return self.save(record)

    def save(self, record: WorkspaceRecord) -> WorkspaceRecord:
        record.updated_at = _now()
        self._path(record.id).write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    def get(self, workspace_id: str) -> WorkspaceRecord:
        path = self._path(workspace_id)
        if not path.exists(): raise KeyError(f"workspace {workspace_id!r} not found")
        return WorkspaceRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[WorkspaceRecord]:
        return sorted([self.get(p.stem) for p in self.root_dir.glob("workspace-*.json")], key=lambda x: x.updated_at, reverse=True)

    def delete(self, workspace_id: str) -> None:
        self._path(workspace_id).unlink(missing_ok=True)

    def _path(self, workspace_id: str) -> Path:
        safe = "".join(c for c in workspace_id if c.isalnum() or c in "-_")
        if not safe: raise ValueError("workspace id is required")
        return self.root_dir / f"{safe}.json"


class WorkspaceToolExecutor:
    SKIP = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}

    def __init__(self, store: WorkspaceStore | None = None) -> None:
        self.store = store or WorkspaceStore()

    def execute(self, adapter: str, arguments: dict[str, Any]) -> dict[str, Any]:
        workspace = self.store.get(str(arguments.get("workspace_id") or ""))
        root = Path(workspace.root_path).resolve()
        if adapter == "workspace_list_files": return {"files": self._list(root, str(arguments.get("path") or ""), int(arguments.get("limit") or 200))}
        if adapter == "workspace_read_file":
            target = self._resolve(root, str(arguments.get("path") or "")); return {"path": self._relative(root, target), "content": target.read_text(encoding="utf-8", errors="replace")[:200_000]}
        if adapter == "workspace_search": return {"matches": self._search(root, str(arguments.get("query") or ""), str(arguments.get("path") or ""), int(arguments.get("limit") or 80))}
        if adapter == "workspace_apply_patch": return self._apply_edit(workspace, root, arguments)
        if adapter == "workspace_write_files": return self._write_files(workspace, root, arguments)
        if adapter == "workspace_git_status": return self._git(root, ["status", "--short"])
        if adapter == "workspace_git_diff": return self._git(root, ["diff", "--", str(arguments.get("path") or "")])
        if adapter == "workspace_run_command": return self._run_command(workspace, root, arguments)
        if adapter == "workspace_create_restore_point": return self._snapshot(root)
        if adapter == "workspace_restore_point": return self._restore(root, str(arguments.get("restore_point_id") or ""))
        raise ValueError(f"unsupported workspace adapter {adapter!r}")

    def _resolve(self, root: Path, value: str) -> Path:
        if not value: raise ValueError("path is required")
        target = (root / value).resolve()
        try: target.relative_to(root)
        except ValueError as exc: raise ValueError("路径必须位于已选工作区内") from exc
        return target

    def _relative(self, root: Path, target: Path) -> str: return str(target.relative_to(root)).replace("\\", "/")
    def _visible(self, relative: Path) -> bool: return not any(part in self.SKIP for part in relative.parts)

    def _list(self, root: Path, raw: str, limit: int) -> list[dict[str, Any]]:
        base = root if not raw else self._resolve(root, raw)
        result=[]
        for item in base.rglob("*"):
            rel=item.relative_to(root)
            if not self._visible(rel): continue
            result.append({"path": self._relative(root,item), "kind": "directory" if item.is_dir() else "file", "size": item.stat().st_size if item.is_file() else 0})
            if len(result)>=max(1,min(limit,1000)): break
        return result

    def _search(self, root: Path, query: str, raw: str, limit: int) -> list[dict[str, Any]]:
        if not query.strip(): raise ValueError("query is required")
        base=root if not raw else self._resolve(root, raw); found=[]; needle=query.lower()
        for item in base.rglob("*"):
            if not item.is_file() or not self._visible(item.relative_to(root)) or item.stat().st_size>1_000_000: continue
            try: lines=item.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError: continue
            for no,line in enumerate(lines,1):
                if needle in line.lower():
                    found.append({"path":self._relative(root,item),"line":no,"text":line[:500]})
                    if len(found)>=max(1,min(limit,500)): return found
        return found

    def _apply_edit(self, workspace: WorkspaceRecord, root: Path, args: dict[str, Any]) -> dict[str, Any]:
        if workspace.read_only: raise PermissionError("该工作区为只读")
        path=self._resolve(root,str(args.get("path") or "")); old=str(args.get("old_text") or ""); new=str(args.get("new_text") or "")
        create=bool(args.get("create",False))
        if create:
            if path.exists(): raise ValueError("文件已存在，不能按 create 创建")
            path.parent.mkdir(parents=True,exist_ok=True); path.write_text(new,encoding="utf-8")
        else:
            if not path.is_file(): raise ValueError("待修改文件不存在")
            content=path.read_text(encoding="utf-8",errors="replace")
            if old not in content: raise ValueError("old_text 未匹配当前文件；请先重新读取文件")
            if content.count(old)!=1: raise ValueError("old_text 匹配多处；请提供更精确的上下文")
            path.write_text(content.replace(old,new,1),encoding="utf-8")
        return {"path":self._relative(root,path),"created":create,"bytes":path.stat().st_size}

    def _write_files(self, workspace: WorkspaceRecord, root: Path, args: dict[str, Any]) -> dict[str, Any]:
        """Create a small project scaffold in one atomic, reviewed operation."""
        if workspace.read_only: raise PermissionError("该工作区为只读")
        files = list(args.get("files") or [])
        if not files or len(files) > 80: raise ValueError("files 必须包含 1 至 80 个文件")
        prepared=[]; total=0
        for item in files:
            if not isinstance(item, dict): raise ValueError("files 中的每一项必须是对象")
            path=self._resolve(root,str(item.get("path") or "")); content=str(item.get("content") or "")
            if path.exists() and not bool(item.get("overwrite", False)): raise ValueError(f"文件已存在：{self._relative(root,path)}")
            total += len(content.encode("utf-8"))
            prepared.append((path, content))
        if total > 2_000_000: raise ValueError("批量写入内容不能超过 2MB")
        for path, content in prepared:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content, encoding="utf-8")
        return {"files":[self._relative(root,path) for path,_ in prepared],"count":len(prepared),"bytes":total}

    def _git(self, root: Path, args: list[str]) -> dict[str, Any]:
        completed=subprocess.run(["git",*args],cwd=root,text=True,capture_output=True,timeout=20,check=False)
        return {"exit_code":completed.returncode,"stdout":completed.stdout[-40_000:],"stderr":completed.stderr[-8_000:]}

    def _run_command(self, workspace: WorkspaceRecord, root: Path, args: dict[str, Any]) -> dict[str, Any]:
        action=str(args.get("action") or "")
        if action not in workspace.allowed_commands: raise PermissionError(f"工作区不允许执行 {action}")
        commands={"pytest":["python","-m","pytest"],"npm_test":["npm","test","--","--runInBand"],"npm_run_build":["npm","run","build"],"npm_run_lint":["npm","run","lint"],"python_compile":["python","-m","compileall"],"javac_compile":["javac","-encoding","UTF-8"]}
        command=[*commands[action]]
        target=str(args.get("target") or "").strip()
        if target:
            safe=self._resolve(root,target); command.append(self._relative(root,safe))
        timeout=max(1,min(int(args.get("timeout_seconds") or 120),600))
        completed=subprocess.run(command,cwd=root,text=True,capture_output=True,timeout=timeout,check=False)
        return {"action":action,"command":command,"exit_code":completed.returncode,"stdout":completed.stdout[-40_000:],"stderr":completed.stderr[-12_000:]}

    def _snapshot(self, root: Path) -> dict[str, Any]:
        point = f"restore-{uuid.uuid4().hex[:12]}"; target = self.store.root_dir / "restore_points" / point
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, target, ignore=shutil.ignore_patterns(*self.SKIP))
        return {"restore_point_id": point, "created_at": _now()}

    def _restore(self, root: Path, point: str) -> dict[str, Any]:
        source = self.store.root_dir / "restore_points" / point
        if not point or not source.is_dir(): raise ValueError("恢复点不存在")
        restored=[]
        for item in source.rglob("*"):
            rel=item.relative_to(source)
            if item.is_file():
                dest=root / rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(item,dest); restored.append(str(rel).replace("\\","/"))
        return {"restore_point_id":point,"restored_files":restored}
