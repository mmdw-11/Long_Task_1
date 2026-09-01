import io
import json
import zipfile

import pytest

from engine.modules.external_imports import openapi_operations, parse_openapi, read_skill_zip, validate_remote_url
from engine.modules.model_connections import ModelConnectionStore


def test_model_connection_store_never_returns_api_key_value(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_SECRET", "do-not-return")
    store = ModelConnectionStore(tmp_path / "models")
    item = store.create({
        "name": "Production model",
        "provider": "openai",
        "model_id": "model-a",
        "base_url": "https://models.example.com/v1",
        "api_key_env": "MODEL_SECRET",
        "tier": "cloud",
    })

    payload = item.to_dict()
    assert payload["api_key_env"] == "MODEL_SECRET"
    assert "do-not-return" not in str(payload)
    assert payload["test_status"] == "untested"


def test_model_connection_accepts_direct_api_key_without_environment_variable(tmp_path):
    store = ModelConnectionStore(tmp_path / "models")
    item = store.create({
        "name": "Direct credential model",
        "provider": "openai-compatible",
        "model_id": "direct-model",
        "base_url": "https://models.example.com/v1",
        "api_key": "direct-secret",
        "tier": "cloud",
    })

    assert store.get(item.id).api_key == "direct-secret"
    assert item.configured is True
    assert item.to_dict()["has_api_key"] is True
    assert "direct-secret" not in str(item.to_dict())


def test_auto_requires_one_runnable_default_for_every_tier(tmp_path):
    store = ModelConnectionStore(tmp_path / "models")
    for tier in ("device", "edge", "cloud"):
        store.create({
            "name": f"{tier} default",
            "provider": "openai-compatible",
            "model_id": f"{tier}-model",
            "base_url": f"https://{tier}.example.com/v1",
            "tier": tier,
            "auto_default": True,
            "test_status": "succeeded",
        })

    status = store.auto_status()

    assert status["ready"] is True
    assert all(status["tiers"][tier]["connection"]["auto_default"] for tier in ("device", "edge", "cloud"))


def test_each_tier_has_only_one_auto_default(tmp_path):
    store = ModelConnectionStore(tmp_path / "models")
    first = store.create({"name":"first", "model_id":"one", "base_url":"https://one.example/v1", "tier":"cloud", "auto_default":True})
    second = store.create({"name":"second", "model_id":"two", "base_url":"https://two.example/v1", "tier":"cloud", "auto_default":True})

    assert store.get(first.id).auto_default is False
    assert store.get(second.id).auto_default is True


def test_openapi_operations_are_split_into_tools():
    document = {
        "openapi": "3.0.0",
        "servers": [{"url": "https://tools.example.com"}],
        "paths": {
            "/weather": {
                "get": {
                    "operationId": "getWeather",
                    "summary": "查询天气",
                    "parameters": [{"name": "city", "in": "query", "required": True, "schema": {"type": "string"}}],
                }
            }
        },
    }
    operations = openapi_operations(parse_openapi(json.dumps(document).encode()), "https://tools.example.com/openapi.json")

    assert operations[0]["name"] == "getWeather"
    assert operations[0]["metadata"]["operation_url"] == "https://tools.example.com/weather"
    assert operations[0]["metadata"]["input_schema"][0]["name"] == "city"


def test_skill_zip_only_imports_safe_text_resources():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("SKILL.md", "# Reliable reviewer\nUse the checklist.")
        archive.writestr("references/checklist.md", "- verify inputs")
        archive.writestr("scripts/install.sh", "echo unsafe")

    imported = read_skill_zip(stream.getvalue())

    assert imported["name"] == "Reliable reviewer"
    assert "references/checklist.md" in imported["references"]
    assert all("scripts/" not in name for name in imported["references"])


def test_skill_zip_rejects_path_traversal():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("SKILL.md", "# Safe")
        archive.writestr("../escape.md", "unsafe")

    with pytest.raises(ValueError, match="路径"):
        read_skill_zip(stream.getvalue())


def test_remote_import_rejects_http_and_private_addresses():
    with pytest.raises(ValueError, match="HTTPS"):
        validate_remote_url("http://example.com/openapi.json")
    with pytest.raises(ValueError, match="私网"):
        validate_remote_url("https://127.0.0.1/openapi.json")
