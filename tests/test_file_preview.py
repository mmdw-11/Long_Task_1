import pytest
from fastapi.testclient import TestClient
from engine.modules.file_preview import preview_run_file
from engine.modules.workspace_tools import WorkspaceStore
from engine.modules.workflows import RunStore
from engine.modules.auth import AuthStore
from engine.server.app import create_app


def setup_workspace(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    store = WorkspaceStore(tmp_path / 'workspaces')
    workspace = store.create(name='demo', root_path=str(root))
    return root, store, workspace


def event(workspace, path, status='succeeded'):
    return {'type': 'tool_result', 'tool_call': {'name': 'workspace_apply_patch', 'status': status,
            'arguments': {'workspace_id': workspace.id}, 'result': {'path': path, 'created': True}}}


def test_current_content_and_no_fake_history(tmp_path):
    root, store, ws = setup_workspace(tmp_path)
    (root / 'main.py').write_bytes('print("你好")\n'.encode('utf8'))
    result = preview_run_file([event(ws, 'main.py')], store, ws.id, 'main.py')
    assert result['content'] == 'print("你好")\n'
    assert result['snapshot'] is False and result['diff_available'] is False


@pytest.mark.parametrize('path', ['../outside', '/outside', 'C:/secret', '.env', 'key.pem'])
def test_rejects_unsafe_paths(tmp_path, path):
    root, store, ws = setup_workspace(tmp_path)
    with pytest.raises((ValueError, PermissionError)):
        preview_run_file([event(ws, path)], store, ws.id, path)


def test_missing_failed_unrecorded_binary_and_large_files(tmp_path):
    root, store, ws = setup_workspace(tmp_path)
    with pytest.raises(PermissionError):
        preview_run_file([event(ws, 'a', 'failed')], store, ws.id, 'a')
    with pytest.raises(PermissionError):
        preview_run_file([event(ws, 'a')], store, ws.id, 'b')
    with pytest.raises(FileNotFoundError):
        preview_run_file([event(ws, 'a')], store, ws.id, 'a')
    for data in [b'\x00binary', b'x' * 200001, b'\xff']:
        (root / 'a').write_bytes(data)
        with pytest.raises(ValueError):
            preview_run_file([event(ws, 'a')], store, ws.id, 'a')


def test_symlink_cannot_escape_workspace(tmp_path):
    root, store, ws = setup_workspace(tmp_path)
    outside = tmp_path / 'outside.txt'
    outside.write_text('private')
    try:
        (root / 'link.txt').symlink_to(outside)
    except OSError:
        pytest.skip('symlinks unavailable on this host')
    with pytest.raises(PermissionError):
        preview_run_file([event(ws, 'link.txt')], store, ws.id, 'link.txt')


def test_api_owner_isolation_and_ownerless_legacy_denied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root, store, ws = setup_workspace(tmp_path)
    (root / 'a.txt').write_text('hello')
    runs = RunStore(tmp_path / 'runs')
    app = create_app(auth_store=AuthStore(tmp_path / 'auth.sqlite3'), auth_required=True, workspace_store=store, run_store=runs)
    alice, bob = TestClient(app), TestClient(app)
    a = alice.post('/api/auth/register', json={'email':'a@example.com','name':'A','password':'password-123'}).json()['user']['id']
    bob.post('/api/auth/register', json={'email':'b@example.com','name':'B','password':'password-123'})
    record = runs.create(input={'input':'test'}, recursion_limit=5)
    record.metadata['owner_user_id'] = a
    record.events = [event(ws, 'a.txt')]
    runs.save(record)
    url = f'/api/runs/{record.id}/files/preview'
    params = {'workspace_id':ws.id,'path':'a.txt'}
    assert alice.get(url, params=params).json()['content'] == 'hello'
    assert bob.get(url, params=params).status_code == 404
    assert TestClient(app).get(url, params=params).status_code == 401
    assert alice.get(url, params={**params,'path':'other'}).status_code == 403
    record.metadata.pop('owner_user_id')
    runs.save(record)
    assert alice.get(url, params=params).status_code == 403
