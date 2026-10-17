"""Process bounding and OCI isolation policy tests."""

import os
import subprocess
import sys

import pytest

from runtime.isolation import BoundedProcessRunner, OCIIsolation, ProcessOutcome
from runtime.secure_tools import ToolExecutor
from runtime.security import SecurityError

PINNED_IMAGE = "example/tool@sha256:" + "a" * 64


def test_bounded_runner_drains_but_does_not_retain_output_flood(tmp_path):
    runner = BoundedProcessRunner(max_output_bytes=4096)
    outcome = runner.run(
        [
            sys.executable,
            "-c",
            "import sys;sys.stdout.write('o'*50000);sys.stderr.write('e'*50000)",
        ],
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", "")},
        timeout=10,
    )
    assert outcome.returncode == 0
    assert len(outcome.stdout.encode()) == 4096
    assert len(outcome.stderr.encode()) == 4096
    assert outcome.stdout_truncated is True
    assert outcome.stderr_truncated is True


def test_bounded_runner_kills_timed_out_process(tmp_path):
    runner = BoundedProcessRunner()
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            cwd=tmp_path,
            env={"PATH": os.environ.get("PATH", "")},
            timeout=1,
        )


def test_oci_command_has_hardening_and_maps_workspace_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "runtime.isolation.shutil.which", lambda *_args, **_kwargs: "/usr/bin/docker"
    )
    target = tmp_path / "input.txt"
    target.write_text("test")
    isolation = OCIIsolation("docker", PINNED_IMAGE, tmp_path, 256)
    invocation = isolation.command(["head", str(target)])
    command = invocation.argv

    assert command[0:3] == ["/usr/bin/docker", "run", "--rm"]
    assert command[3:5] == ["--name", invocation.name]
    assert invocation.name.startswith("agi-tool-")
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges=true" in command
    assert "--pids-limit=16" in command
    assert "--memory=256m" in command
    assert PINNED_IMAGE in command
    assert command[-2:] == ["head", "/workspace/input.txt"]


def test_oci_rejects_unpinned_image_and_outside_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "runtime.isolation.shutil.which", lambda *_args, **_kwargs: "/usr/bin/docker"
    )
    with pytest.raises(SecurityError, match="pinned"):
        OCIIsolation("docker", "example/tool:latest", tmp_path, 256)
    isolation = OCIIsolation("docker", PINNED_IMAGE, tmp_path, 256)
    with pytest.raises(SecurityError, match="outside workspace"):
        isolation.command(["head", "/etc/passwd"])


def test_production_mode_rejects_local_process_execution(tmp_path):
    with pytest.raises(SecurityError, match="require OCI"):
        ToolExecutor(
            workspace=tmp_path,
            allowed_tools={"shell"},
            shell_enabled=True,
            process_isolation="local",
            require_strong_isolation=True,
        )


def test_disabled_and_empty_capability_sets_fail_closed(tmp_path):
    disabled = ToolExecutor(
        workspace=tmp_path,
        allowed_tools={"shell", "python_eval"},
        shell_enabled=True,
        python_enabled=True,
        process_isolation="disabled",
    )
    assert not disabled.allowed_tools
    assert disabled.execute("shell", {"command": "pwd"})["success"] is False

    empty = ToolExecutor(workspace=tmp_path / "empty", allowed_tools=set())
    assert not empty.allowed_tools

    declared = ToolExecutor(
        workspace=tmp_path / "declared",
        allowed_tools={"file_read"},
        python_enabled=True,
        allowed_domains=["example.com"],
    )
    assert declared.allowed_tools == {"file_read"}


def test_oci_timeout_attempts_force_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "runtime.isolation.shutil.which", lambda *_args, **_kwargs: "/usr/bin/docker"
    )
    executor = ToolExecutor(
        workspace=tmp_path,
        allowed_tools={"shell"},
        shell_enabled=True,
        process_isolation="oci",
        oci_image=PINNED_IMAGE,
        require_strong_isolation=True,
        timeout_seconds=1,
    )
    cleaned = []

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    def cleanup(name, _env):
        cleaned.append(name)
        return True

    executor.process_runner.run = timeout
    monkeypatch.setattr(executor.oci, "cleanup", cleanup)
    result = executor.execute("shell", {"command": "pwd"})

    assert result["success"] is False
    assert "Timeout" in result["error"]
    assert len(cleaned) == 1
    assert cleaned[0].startswith("agi-tool-")


def test_real_oci_boundary_when_ci_image_is_provided(tmp_path):
    image = os.environ.get("AGI_OCI_TEST_IMAGE")
    if not image:
        pytest.skip("AGI_OCI_TEST_IMAGE is not configured")
    executor = ToolExecutor(
        workspace=tmp_path,
        allowed_tools={"shell"},
        allowed_commands=["pwd", "touch"],
        shell_enabled=True,
        process_isolation="oci",
        oci_runtime="docker",
        oci_image=image,
        require_strong_isolation=True,
    )

    location = executor.execute("shell", {"command": "pwd"})
    created = executor.execute("shell", {"command": "touch created-by-oci"})

    assert location["success"] is True, location
    assert location["result"].strip() == "/workspace"
    assert created["success"] is True, created
    assert (tmp_path / "created-by-oci").is_file()


def test_tool_executor_routes_process_through_oci_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "runtime.isolation.shutil.which", lambda *_args, **_kwargs: "/usr/bin/docker"
    )
    executor = ToolExecutor(
        workspace=tmp_path,
        allowed_tools={"shell"},
        allowed_commands=["head"],
        shell_enabled=True,
        process_isolation="oci",
        oci_runtime="docker",
        oci_image=PINNED_IMAGE,
        require_strong_isolation=True,
    )
    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed.update(kwargs)
        return ProcessOutcome(0, "safe\n", "", False, False)

    executor.process_runner.run = fake_run
    target = tmp_path / "input.txt"
    target.write_text("safe\n")
    result = executor.execute("shell", {"command": f"head {target}"})

    assert result["success"] is True
    assert result["result"] == "safe\n"
    assert observed["argv"][0] == "/usr/bin/docker"
    assert "--network=none" in observed["argv"]
    assert observed["argv"][-2:] == ["head", "/workspace/input.txt"]
    assert observed["preexec_fn"] is None
