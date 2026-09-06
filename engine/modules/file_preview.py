"""Read-only previews scoped to successful file writes in an authorized Run."""
from pathlib import Path, PurePosixPath
from .workspace_tools import WorkspaceStore

MAX_PREVIEW_BYTES = 200_000


def _sensitive(parts):
    return any(p.lower() in {'.git', '.ssh', '.aws', '.azure', '.venv', 'node_modules'} or p.lower().startswith('.env') for p in parts) or Path(parts[-1]).suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}


def preview_run_file(events, store: WorkspaceStore, workspace_id: str, path: str):
    normalized = path.replace('\\', '/')
    parts = PurePosixPath(normalized).parts
    if not parts or normalized.startswith('/') or any(p in {'.', '..'} or ':' in p for p in parts):
        raise ValueError('文件路径不合法')
    if _sensitive(parts):
        raise PermissionError('敏感文件不提供在线预览')
    permitted = False
    for event in events:
        calls = [event.get('tool_call')] if event.get('type') == 'tool_result' else event.get('tool_calls', []) if event.get('type') == 'node_end' else []
        for call in calls:
            if not isinstance(call, dict) or call.get('status') != 'succeeded':
                continue
            if str((call.get('arguments') or {}).get('workspace_id') or '') != workspace_id:
                continue
            result = call.get('result')
            if not isinstance(result, dict):
                continue
            paths = [result.get('path')] if call.get('name') == 'workspace_apply_patch' else result.get('files', []) if call.get('name') == 'workspace_write_files' else []
            if isinstance(paths, list) and any(isinstance(p, str) and p.replace('\\', '/') == normalized for p in paths):
                permitted = True
    if not permitted:
        raise PermissionError('该文件不在本次运行已确认的编辑记录中')
    root = Path(store.get(workspace_id).root_path).resolve()
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root):
        raise PermissionError('文件路径超出工作区范围')
    if _sensitive(target.relative_to(root).parts):
        raise PermissionError('敏感文件不提供在线预览')
    if not target.is_file():
        raise FileNotFoundError('文件已删除、移动或不再可用')
    with target.open('rb') as stream:
        data = stream.read(MAX_PREVIEW_BYTES + 1)
    if len(data) > MAX_PREVIEW_BYTES:
        raise ValueError('文件超过 200 KB，请在本地编辑器中查看')
    if b'\x00' in data:
        raise ValueError('二进制文件暂不支持预览')
    try:
        content = data.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('文件不是有效的 UTF-8 文本，暂不支持预览') from exc
    return {'path': normalized, 'content': content, 'bytes': len(data), 'snapshot': False,
            'diff_available': False, 'notice': '当前磁盘内容，非运行时快照；本次运行未保存完整修改前后版本。'}
