import asyncio
from pathlib import Path

from examples.context_end_to_end_demo import OUTPUT_ROOT, main


def test_context_end_to_end_demo_outputs_real_artifacts():
    result = asyncio.run(main())

    budget = result["budget_paused"]
    assert budget["run_status"] == "budget_paused"
    assert budget["messages_executed"] is False
    assert "context budget exceeded" in budget["pause_reason"]
    assert Path(budget["memory_md"]).exists()

    validation = result["validation_reroute_resume"]
    assert validation["bad_result_merged"] is False
    assert (
        validation["final_answer"]
        == "Recovered final answer with verified context ledger and checkpoint resume."
    )
    assert validation["resumed_final_answer"] == validation["final_answer"]
    assert validation["resumed_from_step"] == 1
    assert validation["resumed_from_frontier"] == ["draft_answer"]
    assert "requirements_summary" in validation["resumed_state_keys"]
    assert Path(validation["memory_md"]).exists()
    assert Path(validation["resumed_memory_md"]).exists()
    assert Path(validation["checkpoint_final"]).exists()
    assert (OUTPUT_ROOT / "demo_summary.json").exists()
