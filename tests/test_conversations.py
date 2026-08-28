from engine.modules.conversations import ConversationStore, build_conversation_context
from engine.modules.product_ops import normalize_memory_config


def test_conversations_are_isolated_and_clearable(tmp_path):
    store = ConversationStore(tmp_path)
    first = store.create("app-1", "user-1")
    second = store.create("app-1", "user-1")
    first.messages.append({"role": "user", "content": "first"})
    store.save(first)

    assert store.get(second.id).messages == []
    assert store.clear(first.id).messages == []


def test_context_rounds_and_compression(tmp_path):
    store = ConversationStore(tmp_path)
    conversation = store.create("app-1")
    for index in range(10):
        conversation.messages.extend([
            {"role": "user", "content": f"question {index} " + "x" * 500},
            {"role": "assistant", "content": f"answer {index} " + "y" * 500},
        ])

    disabled = build_conversation_context(conversation, normalize_memory_config({"context_rounds": 0}), "now")
    compact = build_conversation_context(conversation, normalize_memory_config({"context_rounds": 8, "context_token_budget": 1600}), "now")

    assert disabled["rounds"] == 0
    assert compact["rounds"] >= 2
    assert compact["compressed"] is True
    assert compact["summary"]
