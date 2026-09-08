"""Zero-API smoke test using the official τ mock state database."""

from tau2.registry import registry
from tau2.runner.build import build_environment


def main() -> None:
    task = next(item for item in registry.get_tasks_loader("mock")(None) if item.id == "create_task_1")
    action = task.evaluation_criteria.actions[0]
    observed = build_environment("mock")
    expected = build_environment("mock")
    before = observed.get_db_hash()
    # The observed path performs an extra valid read. Official state scoring
    # should still accept it because only the final database state matters.
    observed.use_tool("get_users")
    observed.use_tool(action.name, **action.arguments)
    expected.use_tool(action.name, **action.arguments)
    after = observed.get_db_hash()
    assert before != after, "mock write did not mutate state"
    assert after == expected.get_db_hash(), "equivalent correct path did not reach target state"
    print({"mock": "passed", "task": task.id, "initial_hash": before, "final_hash": after})


if __name__ == "__main__":
    main()
