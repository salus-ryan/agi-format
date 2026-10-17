"""Security regression tests for action and network boundaries."""


import pytest

from runtime.secure_tools import ToolExecutor
from runtime.security import (
    NetworkPolicy,
    SecurityError,
    SlidingWindowRateLimiter,
    redact_sensitive,
)


def test_shell_can_be_disabled(tmp_path):
    executor = ToolExecutor(workspace=tmp_path, shell_enabled=False)
    result = executor.execute("shell", {"command": "echo no"})
    assert not result["success"]
    assert "disabled" in result["error"]


def test_shell_rejects_operators_and_non_allowlisted_commands(tmp_path):
    executor = ToolExecutor(workspace=tmp_path)
    assert not executor.execute("shell", {"command": "echo ok; env"})["success"]
    assert not executor.execute("shell", {"command": "env"})["success"]


def test_shell_rejects_paths_outside_workspace(tmp_path):
    executor = ToolExecutor(workspace=tmp_path)
    result = executor.execute("shell", {"command": "head /etc/hosts"})
    assert not result["success"]
    assert "outside workspace" in result["error"]


def test_file_tool_rejects_symlink_escape(tmp_path):
    link = tmp_path / "outside"
    try:
        link.symlink_to("/etc/hosts")
    except OSError:
        pytest.skip("symlinks unavailable")
    executor = ToolExecutor(workspace=tmp_path)
    result = executor.execute("file_read", {"path": str(link)})
    assert not result["success"]
    assert "outside workspace" in result["error"]


def test_shell_rejects_symlink_escape(tmp_path):
    link = tmp_path / "outside"
    try:
        link.symlink_to("/etc/hosts")
    except OSError:
        pytest.skip("symlinks unavailable")
    executor = ToolExecutor(workspace=tmp_path)
    result = executor.execute("shell", {"command": "head outside"})
    assert not result["success"]
    assert "outside workspace" in result["error"]


def test_subprocess_environment_does_not_copy_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("VERY_SECRET_VALUE", "must-not-leak")
    executor = ToolExecutor(workspace=tmp_path)
    environment = executor._minimal_env()
    assert "VERY_SECRET_VALUE" not in environment
    assert set(environment) == {"PATH", "HOME", "LANG", "LC_ALL"}


def test_network_is_deny_by_default():
    with pytest.raises(SecurityError, match="disabled"):
        NetworkPolicy().validate_url("https://example.com")


def test_network_rejects_private_targets_even_when_allowlisted():
    with pytest.raises(SecurityError, match="Private"):
        NetworkPolicy(("localhost",)).validate_url("https://localhost")


def test_secret_redaction_handles_keys_and_inline_values():
    redacted = redact_sensitive(
        {"api_key": "abc", "message": "password=hunter2", "safe": "value"}
    )
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["message"] == "password=[REDACTED]"
    assert redacted["safe"] == "value"


def test_rate_limiter_enforces_sliding_window():
    limiter = SlidingWindowRateLimiter(2, 10)
    assert limiter.check("user", now=0)[0]
    assert limiter.check("user", now=1)[0]
    allowed, retry_after = limiter.check("user", now=2)
    assert not allowed
    assert retry_after > 0
    assert limiter.check("user", now=11)[0]
