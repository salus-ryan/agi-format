"""Security regression tests for action and network boundaries."""

import socket

import httpx
import pytest

from runtime.secure_tools import ToolExecutor
from runtime.security import NetworkPolicy, SecurityError

PUBLIC_IP = "93.184.215.14"


def fake_dns(monkeypatch, *addresses):
    def getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in addresses]

    monkeypatch.setattr("runtime.security.socket.getaddrinfo", getaddrinfo)


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


@pytest.mark.parametrize(
    "command",
    [
        "wc --files0-from=/etc/hostname",
        "sort --files0-from=/etc/passwd",
        "wc --files0-from /etc/hostname",
        "head --bytes=/etc/hostname",
        "sort -o/etc/agi-test-output a",
        "sort --output=/tmp/agi-test-output a",
        "head ~/.ssh/id_rsa",
    ],
)
def test_shell_rejects_option_values_that_escape_the_workspace(tmp_path, command):
    executor = ToolExecutor(workspace=tmp_path, allowed_commands=["wc", "sort", "head"])
    result = executor.execute("shell", {"command": command})
    assert not result["success"]
    assert "stderr" not in result


def test_network_pins_request_to_the_validated_address(tmp_path, monkeypatch):
    fake_dns(monkeypatch, PUBLIC_IP)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        seen["header"] = request.headers["host"]
        seen["sni"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, text="ok")

    executor = ToolExecutor(
        workspace=tmp_path, allowed_tools={"web_request"}, allowed_domains=["example.com"]
    )
    executor._transport = httpx.MockTransport(handler)
    result = executor.execute("web_request", {"url": "https://api.example.com/v1"})
    assert result["success"], result
    assert seen == {"host": PUBLIC_IP, "header": "api.example.com", "sni": "api.example.com"}


def test_network_rejects_any_private_answer(monkeypatch):
    fake_dns(monkeypatch, PUBLIC_IP, "169.254.169.254")
    with pytest.raises(SecurityError, match="Private"):
        NetworkPolicy(("example.com",)).validate_url("https://example.com")


def test_network_post_is_rejected_before_resolution(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("DNS must not be queried")

    monkeypatch.setattr("runtime.security.socket.getaddrinfo", fail)
    executor = ToolExecutor(
        workspace=tmp_path, allowed_tools={"web_request"}, allowed_domains=["example.com"]
    )
    result = executor.execute("web_request", {"url": "https://example.com", "method": "POST"})
    assert result == {"success": False, "error": "Only GET and HEAD are enabled",
                      "elapsed_ms": result["elapsed_ms"]}
