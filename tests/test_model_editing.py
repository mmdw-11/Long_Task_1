from engine.modules.model_connections import ModelConnection, ModelConnectionStore


def test_rename_preserves_identity_key_and_test_status(tmp_path):
    store = ModelConnectionStore(tmp_path)
    original = store.create({"name": "Before", "model_id": "chat", "base_url": "https://example.com/v1", "api_key": "test-secret", "test_status": "succeeded"})
    renamed = store.save(ModelConnection.from_dict({**original.to_storage_dict(), "name": "After"}))
    assert renamed.id == original.id
    assert renamed.api_key == "test-secret"
    assert renamed.test_status == "succeeded"
    assert "api_key" not in renamed.to_dict()


def test_connection_changes_require_retest_and_default_is_unique(tmp_path):
    store = ModelConnectionStore(tmp_path)
    first = store.create({"name": "One", "model_id": "chat", "base_url": "https://example.com/v1", "auto_default": True, "test_status": "succeeded"})
    second = store.create({"name": "Two", "model_id": "chat", "base_url": "https://example.com/v1", "auto_default": True})
    assert not store.get(first.id).auto_default
    assert store.get(second.id).auto_default
    changed = store.save(ModelConnection.from_dict({**first.to_storage_dict(), "api_key": "replacement"}))
    assert changed.test_status == "untested"
