import json

from context_manager import ContextWindowManager


def manager(tmp_path, *, limit=5000):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return ContextWindowManager(
        context_limit=limit,
        max_output_tokens=500,
        workspace=workspace,
        audit_path=tmp_path / "context-compactions.jsonl",
        input_safety_tokens=100,
        recent_messages=6,
    )


def test_cjk_estimate_is_conservative_and_compacts_with_checkpoint(tmp_path):
    subject = manager(tmp_path, limit=1800)
    messages = [
        {"role": "system", "content": "Do work."},
        {"role": "user", "content": "处理合同并验证输出。"},
    ]
    for index in range(18):
        messages.extend([
            {"role": "assistant", "content": None, "reasoning": f"分析步骤 {index} " + "字" * 200,
             "tool_calls": [{"id": f"call-{index}", "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"call-{index}", "content": f"结果 {index} " + "行" * 300},
        ])
    compacted = subject.prepare(messages)
    assert len(compacted) < len(messages)
    assert subject.estimate_tokens(compacted) <= subject.input_budget
    assert compacted[0] == messages[0]
    assert compacted[1] == messages[1]
    assert compacted[-1]["content"].startswith("结果 17")
    checkpoint = subject.workspace / ".cowork_context/checkpoint.md"
    assert checkpoint.exists()
    assert "字" in checkpoint.read_text(encoding="utf-8")
    audit = json.loads((tmp_path / "context-compactions.jsonl").read_text(encoding="utf-8"))
    assert audit["after_messages"] == len(compacted)


def test_short_history_is_left_unchanged(tmp_path):
    subject = manager(tmp_path)
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    assert subject.prepare(messages) is messages
    assert subject.compactions == 0


def test_reasoning_and_tool_calls_count_toward_context(tmp_path):
    subject = manager(tmp_path)
    content_only = [{"role": "assistant", "content": "ok"}]
    with_reasoning = [{"role": "assistant", "content": "ok", "reasoning": "推理过程" * 50,
                       "tool_calls": [{"id": "x", "function": {"name": "bash", "arguments": "python " + "x" * 200}}]}]
    assert subject.estimate_tokens(with_reasoning) > subject.estimate_tokens(content_only)
