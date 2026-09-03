from pathlib import Path

import pytest

from engine.modules.workspace_tools import WorkspaceStore, WorkspaceToolExecutor


def test_workspace_tools_read_search_write_and_escape_protection(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("def hello():\n    return 'old'\n", encoding="utf-8")
    store = WorkspaceStore(tmp_path / "workspaces")
    workspace = store.create(name="demo", root_path=str(root))
    tools = WorkspaceToolExecutor(store)
    args = {"workspace_id": workspace.id}
    assert tools.execute("workspace_list_files", args)["files"][0]["path"] == "app.py"
    assert "old" in tools.execute("workspace_read_file", {**args, "path": "app.py"})["content"]
    assert tools.execute("workspace_search", {**args, "query": "hello"})["matches"][0]["line"] == 1
    result = tools.execute("workspace_apply_patch", {**args, "path": "app.py", "old_text": "'old'", "new_text": "'new'"})
    assert result["path"] == "app.py"
    assert "new" in (root / "app.py").read_text(encoding="utf-8")
    restore_point = tools.execute("workspace_create_restore_point", args)["restore_point_id"]
    tools.execute("workspace_apply_patch", {**args, "path": "app.py", "old_text": "'new'", "new_text": "'later'"})
    tools.execute("workspace_restore_point", {**args, "restore_point_id": restore_point})
    assert "new" in (root / "app.py").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="工作区"):
        tools.execute("workspace_read_file", {**args, "path": "../outside.txt"})


def test_workspace_read_only_and_command_allowlist(tmp_path: Path):
    root = tmp_path / "project"; root.mkdir()
    (root / "a.py").write_text("x = 1", encoding="utf-8")
    store = WorkspaceStore(tmp_path / "workspaces")
    workspace = store.create(name="readonly", root_path=str(root), read_only=True, allowed_commands=["python_compile"])
    tools = WorkspaceToolExecutor(store)
    with pytest.raises(PermissionError):
        tools.execute("workspace_apply_patch", {"workspace_id": workspace.id, "path": "a.py", "old_text": "1", "new_text": "2"})
    with pytest.raises(PermissionError):
        tools.execute("workspace_run_command", {"workspace_id": workspace.id, "action": "pytest"})
