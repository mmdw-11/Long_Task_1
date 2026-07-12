from engine import (
    AdaptiveResourceScheduler,
    END,
    SensitiveDataRedactor,
    StateGraph,
)
from engine.hooks import (
    AUDIT_PACK_KEY,
    HookManager,
    REDACTION_RESULT_KEY,
    RESOURCE_ALLOCATION_KEY,
)


def test_redactor_masks_secrets_without_storing_raw_values():
    redactor = SensitiveDataRedactor()
    result = redactor.redact(
        {
            "text": "api_key=sk-testSECRET123456 user alice@example.com phone 13800138000",
        }
    )

    redacted = result.redacted["text"]
    assert "sk-testSECRET123456" not in redacted
    assert "alice@example.com" not in redacted
    assert "13800138000" not in redacted
    assert "[REDACTED:API_KEY:1]" in redacted
    assert {finding.kind for finding in result.findings} >= {"api_key", "email", "phone"}
    assert all("SECRET" not in str(finding.to_dict()) for finding in result.findings)


def test_audit_pack_does_not_include_raw_secret():
    from engine import AuditPackBuilder

    redactor = SensitiveDataRedactor()
    result = redactor.redact("token=abc12345678901234567890")
    pack = AuditPackBuilder().build(node="n", result=result)
    data = pack.to_dict()

    assert data["original_hash"] != data["redacted_hash"]
    assert "abc12345678901234567890" not in str(data)
    assert data["findings_summary"]["kinds"]["token"] == 1


def test_hook_injects_redaction_and_audit_for_sensitive_scheduled_node():
    graph = StateGraph()

    def worker(state):
        return {
            "redaction": state[REDACTION_RESULT_KEY],
            "audit": state[AUDIT_PACK_KEY],
            "allocation": state[RESOURCE_ALLOCATION_KEY],
        }

    graph.add_node(
        "worker",
        worker,
        metadata={
            "sensitivity": "secret",
            "complexity": "high",
            "description": "处理 api_key=sk-prodSECRET123456 的安全事件",
        },
    )
    graph.set_entry_point("worker")
    graph.add_edge("worker", END)

    compiled = graph.compile()
    compiled.hooks = HookManager(scheduler=AdaptiveResourceScheduler())
    state = compiled.invoke({"input": "token=abc12345678901234567890"})

    assert state["redaction"]["summary"]["redacted_count"] >= 2
    assert "abc12345678901234567890" not in str(state["audit"])
    assert state["audit"]["findings_summary"]["redacted_count"] >= 2
    assert "redaction" in state["allocation"]["metadata"]
    assert "audit_pack" in state["allocation"]["metadata"]
