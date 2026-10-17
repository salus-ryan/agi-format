"""Regression tests for the full asynchronous kernel step."""

from unittest.mock import AsyncMock

import pytest

from runtime.live_modules import LiveKernel
from runtime.loader import load_agi
from runtime.types import AlignmentCheck


@pytest.mark.asyncio
async def test_step_computes_reward_before_goal_update(tmp_path, monkeypatch):
    monkeypatch.setenv("AGI_ENABLE_ONLINE_TRAINING", "false")
    package = load_agi("skeleton")
    kernel = LiveKernel(
        package,
        vllm_base_url="http://127.0.0.1:1",
        model_name="test-model",
        model_revision="test-revision",
        memory_path=str(tmp_path / "memory"),
    )

    assert "shell" not in kernel.available_tools
    assert "python_eval" not in kernel.available_tools
    assert "web_request" not in kernel.available_tools

    kernel._ground_input = AsyncMock(return_value=[])
    kernel.knowledge_graph.extract_facts_from_text = AsyncMock(return_value=[])
    kernel.memory_store.search_async = AsyncMock(return_value=[])
    kernel.memory_store.add_async = AsyncMock(return_value=None)
    kernel.planner.search = AsyncMock(return_value=None)
    kernel.llm.chat_async = AsyncMock(return_value="hello")
    kernel._check_alignment_async = AsyncMock(
        return_value=AlignmentCheck(approved=True, violated_constraints=[], explanation="OK")
    )
    kernel._update_user_model = AsyncMock(return_value=None)
    kernel._update_goals = AsyncMock(return_value=None)

    result = await kernel.step_async("hello API_KEY=topsecret", max_tokens=16, temperature=0)

    assert result["response"] == "hello"
    persisted_text = kernel.memory_store.add_async.await_args.args[0]
    assert "topsecret" not in persisted_text
    assert "[REDACTED]" in persisted_text
    reward = kernel._update_goals.await_args.args[2]
    assert isinstance(reward, float)
    assert result["step_log"]["events"][-1]["phase"] == "learn"
    await kernel.close()
