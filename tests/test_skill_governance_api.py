"""技能治理接口测试。

覆盖技能版本历史、灰度比例和回滚链路，确保前端后续只调用 REST 接口即可完成治理闭环。
"""

from fastapi.testclient import TestClient

from engine.modules.skills import SkillRepository, SkillStatus
from engine.server.app import create_app


def test_skill_versions_rollout_and_rollback_api(tmp_path):
    # 预置一个已发布技能，模拟审批完成后进入线上检索池的状态。
    repository = SkillRepository(tmp_path / "skills")
    repository.create(
        skill_id="meeting-summary",
        name="会议复盘技能",
        description="用于会议总结和行动项提取",
        tags=["会议", "复盘"],
        status=SkillStatus.PUBLISHED,
        metadata={"rollout_percent": 100},
        source_run_ids=["run-source"],
        content="\n".join(
            [
                "# 会议复盘技能",
                "",
                "## 适用场景",
                "- 用户需要整理会议纪要、行动项和负责人。",
                "",
                "## 执行步骤",
                "1. 先提取会议目标、关键决策和待办事项。",
                "2. 再为每个待办事项补充负责人、截止时间和风险。",
                "",
                "## 校验方式",
                "- 输出必须包含会议结论、行动项、负责人和下一步。",
            ]
        ),
    )
    client = TestClient(create_app(skill_repository=repository))

    # 灰度比例设置为 0 后，技能仍是 published，但不会被检索注入。
    closed = client.post(
        "/api/skills/meeting-summary/rollout",
        json={"percent": 0, "approved_by": "pm"},
    )
    assert closed.status_code == 200
    # 灰度属于运行元数据，不应制造新的内容发布版本。
    assert closed.json()["version"] == 1
    assert closed.json()["metadata"]["rollout_percent"] == 0

    search_closed = client.post(
        "/api/skills/search",
        json={"query": "请整理会议复盘和行动项", "node": "writer"},
    )
    assert search_closed.status_code == 200
    assert search_closed.json()["matches"] == []

    versions = client.get("/api/skills/meeting-summary/versions")
    assert versions.status_code == 200
    assert [item["version"] for item in versions.json()] == [1]

    old_version = client.get("/api/skills/meeting-summary/versions/1")
    assert old_version.status_code == 200
    assert old_version.json()["metadata"]["rollout_percent"] == 0

    reopened = client.post(
        "/api/skills/meeting-summary/rollout",
        json={"percent": 100, "approved_by": "pm"},
    )
    assert reopened.status_code == 200
    assert reopened.json()["version"] == 1

    search_open = client.post(
        "/api/skills/search",
        json={"query": "请整理会议复盘和行动项", "node": "writer"},
    )
    assert search_open.status_code == 200
    assert search_open.json()["matches"][0]["skill"]["id"] == "meeting-summary"


def test_skill_governance_requires_approver(tmp_path):
    # 关键治理动作必须有审批人，避免后台静默发布或回滚。
    repository = SkillRepository(tmp_path / "skills")
    repository.create(
        skill_id="safe-skill",
        name="安全技能",
        status=SkillStatus.PUBLISHED,
        content="# 安全技能\n\n## 适用场景\n- 安全处理。\n\n## 执行步骤\n1. 检查输入。\n\n## 校验方式\n- 记录结果。",
    )
    client = TestClient(create_app(skill_repository=repository))

    rollout = client.post("/api/skills/safe-skill/rollout", json={"percent": 50, "approved_by": ""})
    assert rollout.status_code == 400

    rollback = client.post("/api/skills/safe-skill/rollback/1", json={})
    assert rollback.status_code == 400
