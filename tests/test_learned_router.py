from engine import (
    AdaptiveResourceScheduler,
    BinaryTextRouterModel,
    BgeM3Encoder,
    RealCascadeTeacher,
    LearnedTaskGate,
    PseudoCascadeTeacher,
    ResourceRequest,
    ResourceTier,
    build_route_dataset,
    build_route_dataset_from_requests,
    render_experiment_report,
    save_experiment_report,
    train_router,
)


def test_build_route_dataset_from_teacher():
    texts = [
        "帮我总结这封邮件",
        "请分析这份长文档的复杂系统设计",
    ]
    dataset = build_route_dataset(texts, teacher=PseudoCascadeTeacher())

    assert len(dataset.examples) == 2
    assert {ex.label for ex in dataset.examples}.issubset({0, 1})


def test_build_route_dataset_from_requests():
    requests = [
        ResourceRequest(node="n1", state={"input": "简单待办"}),
        ResourceRequest(node="n2", state={"input": "复杂代码重构"}),
    ]

    dataset = build_route_dataset_from_requests(requests)

    assert len(dataset.examples) == 2
    assert all(ex.text for ex in dataset.examples)


def test_binary_router_trains_and_predicts():
    dataset = build_route_dataset(
        [
            "简单待办事项",
            "总结一下会议纪要",
            "复杂代码重构和性能分析",
            "长文档和图像内容理解",
        ]
    )
    result = train_router(dataset, train_ratio=0.5)
    model = BinaryTextRouterModel().fit(dataset.examples)

    assert result["metrics"]["f1"] >= 0.0
    assert model.predict("复杂代码重构和性能分析") == 1
    assert model.predict("总结邮件要点") in {0, 1}


def test_learned_gate_integrates_with_scheduler():
    dataset = build_route_dataset(
        [
            "总结邮件",
            "简单待办",
            "复杂代码重构",
            "大型文档分析",
        ]
    )
    model = train_router(dataset, train_ratio=0.5)["model"]
    gate = LearnedTaskGate(model)
    scheduler = AdaptiveResourceScheduler(gate=gate)

    allocation = scheduler.acquire(
        ResourceRequest(node="route", state={"input": "复杂代码重构和架构分析"})
    )

    assert allocation.metadata["decision"]["profile"]["metadata"]["router"] == "learned"
    assert allocation.tier in {ResourceTier.CLOUD, ResourceTier.EDGE}


def test_render_experiment_report_and_save(tmp_path):
    text = render_experiment_report(
        [
            {"method": "rule", "accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65, "notes": "baseline"},
            {"method": "learned", "accuracy": 0.9, "precision": 0.85, "recall": 0.8, "f1": 0.82, "notes": "mixed n-gram"},
        ]
    )
    path = save_experiment_report([{"method": "rule", "accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65}], tmp_path / "report.md")

    assert "| method | accuracy | precision | recall | f1 | notes |" in text
    assert path.exists()


def test_bge_m3_encoder_has_clear_missing_dependency_error():
    encoder = BgeM3Encoder()
    try:
        encoder.encode("hello")
    except ImportError as exc:
        assert "FlagEmbedding" in str(exc)
    else:
        raise AssertionError("expected ImportError when FlagEmbedding is unavailable")


def test_real_cascade_teacher_runs_true_small_judge_large_chain():
    class _Message:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Message(content)

    class _Response:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Completions:
        def __init__(self):
            self.calls = []

        def create(self, *, model, messages, temperature):
            self.calls.append({"model": model, "messages": messages, "temperature": temperature})
            user_text = messages[-1]["content"]
            if "Return JSON with keys: use_large" in user_text:
                return _Response('{"use_large": true, "confidence": 0.92, "reason": "task needs deeper reasoning"}')
            if "Produce the stronger final answer." in user_text:
                return _Response("large answer")
            return _Response("small answer")

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    teacher = RealCascadeTeacher(
        client=_Client(),
        small_model="small-model",
        judge_model="judge-model",
        large_model="large-model",
    )
    dataset = build_route_dataset(
        ["请分析这段复杂代码并指出潜在架构风险"],
        teacher=teacher,
    )

    assert len(dataset.examples) == 1
    example = dataset.examples[0]
    assert example.label == 1
    assert example.metadata["cascade"]["promoted"] is True
    assert example.metadata["cascade"]["small_output"] == "small answer"
    assert example.metadata["cascade"]["large_output"] == "large answer"
