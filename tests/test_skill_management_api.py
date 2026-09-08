from fastapi.testclient import TestClient

from engine.modules.auth import AuthStore
from engine.modules.product_ops import ApplicationStore
from engine.modules.skills import SkillRepository
from engine.modules.skills import SkillSemanticIndex, SkillRetriever, SkillStatus
from engine.modules.workflows import WorkflowStore
from engine.server.app import create_app


CONTENT = """# 测试 Skill

## 适用场景
用于验证 Skill 管理流程和筛选行为。

## 执行步骤
1. 读取输入。
2. 返回结构化结果。

## 校验方式
检查结果是否包含输入和结论。
"""


def client(tmp_path):
    return TestClient(create_app(
        skill_repository=SkillRepository(tmp_path / "skills"),
        application_store=ApplicationStore(tmp_path / "apps"),
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        auth_store=AuthStore(tmp_path / "auth.sqlite3"),
    ))


def test_install_is_per_user_and_idempotent(tmp_path):
    c = client(tmp_path)
    market = c.get("/api/marketplace/skills").json()
    item = market[0]
    assert item["installed"] is False
    first = c.post(f"/api/marketplace/skills/{item['slug']}/install")
    second = c.post(f"/api/marketplace/skills/{item['slug']}/install")
    assert first.json()["installed"] is True
    assert second.json()["installed"] is False
    assert c.get("/api/skills?scope=installed").json()[0]["installed"] is True
    assert c.delete(f"/api/marketplace/skills/{item['slug']}/install").status_code == 200
    assert c.get("/api/skills?scope=installed").json() == []


def test_private_skill_draft_validate_publish_and_filter(tmp_path):
    c = client(tmp_path)
    created = c.post("/api/skills", json={
        "name": "筛选测试", "description": "用于会议总结", "content": CONTENT,
        "tags": ["office", "meeting", "testtag"],
        "metadata": {"category": "office", "scenarios": ["meeting"], "keywords": ["testtag"]},
    }).json()
    assert created["status"] == "draft"
    assert created["editable"] is True
    assert c.get("/api/skills?scope=mine&query=testtag&category=office").json()[0]["id"] == created["id"]
    report = c.post(f"/api/skills/{created['id']}/validate").json()
    assert report["passed"] is True
    published = c.post(f"/api/skills/{created['id']}/publish", json={}).json()
    assert published["status"] == "published"
    assert published["version"] == 2

    revision = c.put(f"/api/skills/{created['id']}", json={
        "name": "筛选测试", "description": "new description", "content": CONTENT + "\nnew content",
        "tags": ["office"], "metadata": {"category": "office"},
    }).json()
    assert revision["id"] != created["id"]
    assert revision["status"] == "draft"
    assert c.get(f"/api/skills/{created['id']}").json()["status"] == "published"
    repeated_revision = c.put(f"/api/skills/{created['id']}", json={
        "name": "筛选测试", "description": "new description", "content": CONTENT + "\nnew content",
        "tags": ["office"], "metadata": {"category": "office"},
    }).json()
    assert repeated_revision["id"] == revision["id"]
    assert c.post(f"/api/skills/{revision['id']}/validate").json()["passed"] is True
    updated = c.post(f"/api/skills/{revision['id']}/publish", json={}).json()
    assert updated["id"] == created["id"]
    assert updated["version"] == 3
    assert updated["description"] == "new description"


def test_builtin_is_read_only_and_copy_becomes_draft(tmp_path):
    c = client(tmp_path)
    builtin = c.get("/api/skills").json()[0]
    assert builtin["visibility"] == "builtin"
    assert builtin["editable"] is False
    assert c.put(f"/api/skills/{builtin['id']}", json={"name":"x","content":CONTENT}).status_code == 403
    copied = c.post(f"/api/skills/{builtin['id']}/copy").json()
    assert copied["visibility"] == "private"
    assert copied["status"] == "draft"
    assert copied["editable"] is True


def test_retrieval_score_is_not_diluted_by_long_skill_body(tmp_path):
    c = client(tmp_path)
    market = c.get("/api/marketplace/skills").json()
    email = next(item for item in market if item["slug"] == "email-writer")
    c.post("/api/marketplace/skills/email-writer/install")
    result = c.post("/api/skills/search", json={"query": "帮我写一个商务上的邮件回复模板", "top_k": 5}).json()
    matched = next(item for item in result["matches"] if item["skill_id"] == email["skill_id"])
    assert matched["score"] >= 0.5
    assert matched["reason"].startswith("命中关键词：")


def test_retrieval_understands_common_business_paraphrases(tmp_path):
    c = client(tmp_path)
    market = c.get("/api/marketplace/skills").json()
    travel = next(item for item in market if item["slug"] == "travel-planner")
    c.post("/api/marketplace/skills/travel-planner/install")
    result = c.post("/api/skills/search", json={"query": "我想去旅游", "top_k": 5}).json()
    assert result["matches"]
    assert result["matches"][0]["skill_id"] == travel["skill_id"]
    assert result["matches"][0]["score"] >= 0.5


def test_semantic_index_and_hybrid_score_breakdown(tmp_path):
    class FakeSemanticEmbedding:
        def embed(self, text):
            return [1.0, 0.0] if any(word in text for word in ("散心", "旅行", "旅游")) else [0.0, 1.0]

    repository = SkillRepository(tmp_path / "semantic-skills")
    travel = repository.create(skill_id="travel", name="旅行计划", description="规划行程", content=CONTENT,
                               status=SkillStatus.PUBLISHED, tags=["旅行"])
    index = SkillSemanticIndex(tmp_path / "skill-index.sqlite3", embedding_model="fake", embedder=FakeSemanticEmbedding())
    assert index.upsert(travel)["status"] == "ready"
    result = SkillRetriever(repository, semantic_index=index).retrieve_detailed("最近压力很大，想出去散心", mode="hybrid")
    assert result["degraded"] is False
    assert result["results"][0]["match"].skill.id == "travel"
    assert result["results"][0]["semantic_score"] == 1.0
    assert result["results"][0]["match"].score >= 0.65


def test_hybrid_keeps_semantic_hit_without_literal_overlap(tmp_path):
    class ModerateSemanticEmbedding:
        def embed(self, text):
            if "想出去玩" in text:
                return [1.0, 0.0]
            return [0.55, 0.835164654]

    repository = SkillRepository(tmp_path / "moderate-semantic-skills")
    travel = repository.create(skill_id="travel", name="休闲助手", description="提供放松建议", content=CONTENT,
                               status=SkillStatus.PUBLISHED, tags=["休闲"])
    index = SkillSemanticIndex(tmp_path / "moderate-index.sqlite3", embedding_model="fake", embedder=ModerateSemanticEmbedding())
    index.upsert(travel)
    result = SkillRetriever(repository, semantic_index=index).retrieve_detailed("我想出去玩", mode="hybrid")
    assert result["results"][0]["match"].skill.id == "travel"
    assert result["results"][0]["semantic_score"] == 0.55
    assert result["results"][0]["match"].score == 0.55
