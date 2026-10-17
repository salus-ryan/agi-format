"""Deny-by-default tool execution for the reference runtime.

This is defense in depth, not a replacement for a container or microVM boundary.
Production operators should keep shell and Python disabled unless the whole agent
process is itself isolated from credentials and sensitive networks.
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

from runtime.isolation import BoundedProcessRunner, OCIIsolation
from runtime.security import NetworkPolicy, SecurityError

log = structlog.get_logger()


class ToolExecutor:
    """Execute a small set of tools under an explicit capability policy."""

    DEFAULT_COMMANDS = ("pwd", "ls", "echo", "printf", "wc", "head", "tail")
    DEFAULT_TOOLS = frozenset({"shell", "file_read", "file_write", "communicate"})
    MAX_OUTPUT = 10_000
    MAX_FILE_BYTES = 5 * 1024 * 1024
    MAX_COMMAND_CHARS = 2_000
    BLOCKED_SHELL_SYNTAX = re.compile(r"[;&|<>`\n\r]|\$\(|\$\{")

    def __init__(
        self,
        workspace: str | Path | None = None,
        *,
        allowed_tools: set[str] | None = None,
        allowed_commands: list[str] | tuple[str, ...] | None = None,
        allowed_domains: list[str] | tuple[str, ...] | None = None,
        shell_enabled: bool = True,
        python_enabled: bool = False,
        network_post_enabled: bool = False,
        timeout_seconds: int = 30,
        max_memory_mb: int = 512,
        process_isolation: str = "local",
        oci_runtime: str = "docker",
        oci_image: str = "",
        require_strong_isolation: bool = False,
    ):
        default_workspace = Path(tempfile.gettempdir()) / "agi-agent-workspace"
        self.workspace = Path(workspace or os.environ.get("AGI_WORKSPACE", default_workspace)).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        explicit_tools = allowed_tools is not None
        self.allowed_tools = set(self.DEFAULT_TOOLS if allowed_tools is None else allowed_tools)
        if not shell_enabled:
            self.allowed_tools.discard("shell")
        if python_enabled and not explicit_tools:
            self.allowed_tools.add("python_eval")
        elif not python_enabled:
            self.allowed_tools.discard("python_eval")
        process_isolation = process_isolation.lower()
        if process_isolation not in {"disabled", "local", "oci"}:
            raise SecurityError("Process isolation must be disabled, local, or oci")
        process_tools_enabled = bool(self.allowed_tools & {"shell", "python_eval"})
        if process_isolation == "disabled":
            self.allowed_tools.difference_update({"shell", "python_eval"})
            process_tools_enabled = False
        if require_strong_isolation and process_tools_enabled and process_isolation != "oci":
            raise SecurityError("Process tools require OCI isolation in production mode")
        if allowed_domains and not explicit_tools:
            self.allowed_tools.add("web_request")
        elif not allowed_domains:
            self.allowed_tools.discard("web_request")
        self.allowed_commands = frozenset(
            self.DEFAULT_COMMANDS if allowed_commands is None else allowed_commands
        )
        self.network_policy = NetworkPolicy(tuple(domain.lower() for domain in (allowed_domains or ())))
        self.network_post_enabled = network_post_enabled
        self.timeout_seconds = max(1, min(int(timeout_seconds), 60))
        self.max_memory_mb = max(64, min(int(max_memory_mb), 2048))
        self.process_isolation = process_isolation
        self.process_runner = BoundedProcessRunner(self.MAX_OUTPUT)
        self.oci = None
        if process_isolation == "oci" and process_tools_enabled:
            self.oci = OCIIsolation(
                runtime=oci_runtime,
                image=oci_image,
                workspace=self.workspace,
                max_memory_mb=self.max_memory_mb,
            )
        self.execution_count = 0
        self.success_count = 0

    def execute(self, tool: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute an allowed tool and return a bounded, serializable result."""
        self.execution_count += 1
        start = time.monotonic()
        if tool not in self.allowed_tools:
            return self._finish(
                {"success": False, "error": f"Tool is disabled by policy: {tool}"}, start
            )
        try:
            if tool == "shell":
                result = self._exec_shell(str(params.get("command", "")))
            elif tool == "file_read":
                result = self._exec_file_read(str(params.get("path", "")))
            elif tool == "file_write":
                result = self._exec_file_write(
                    str(params.get("path", "")), str(params.get("content", ""))
                )
            elif tool == "web_request":
                result = self._exec_web_request(
                    str(params.get("url", "")),
                    str(params.get("method", "GET")),
                    params.get("body"),
                )
            elif tool == "python_eval":
                result = self._exec_python(str(params.get("code", "")))
            elif tool == "communicate":
                result = {"success": True, "result": str(params.get("message", ""))}
            else:
                result = {"success": False, "error": f"Unknown tool: {tool}"}
        except (OSError, ValueError, SecurityError, subprocess.SubprocessError) as exc:
            result = {"success": False, "error": str(exc)}
        except Exception:
            log.exception("tool_execution_failed", tool=tool)
            result = {"success": False, "error": "Tool execution failed"}
        return self._finish(result, start)

    def _finish(self, result: dict[str, Any], start: float) -> dict[str, Any]:
        if result.get("success"):
            self.success_count += 1
        result["elapsed_ms"] = round((time.monotonic() - start) * 1000, 1)
        log.info("tool_executed", success=bool(result.get("success")))
        return result

    def _minimal_env(self) -> dict[str, str]:
        executable_dir = str(Path(sys.executable).resolve().parent)
        safe_path = os.pathsep.join(dict.fromkeys([executable_dir, *os.defpath.split(os.pathsep)]))
        return {
            "PATH": safe_path,
            "HOME": str(self.workspace),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }

    def _resource_limits(self):
        if os.name != "posix" or platform.system() == "Android":
            return None

        def apply_limits() -> None:
            import resource

            memory = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout_seconds, self.timeout_seconds + 1))
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            resource.setrlimit(resource.RLIMIT_FSIZE, (self.MAX_FILE_BYTES, self.MAX_FILE_BYTES))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))

        return apply_limits

    def _run_process(self, argv: list[str]) -> dict[str, Any]:
        if self.process_isolation == "disabled":
            return {"success": False, "error": "Process execution is disabled"}
        invocation = self.oci.command(argv) if self.oci else None
        command = invocation.argv if invocation else argv
        try:
            completed = self.process_runner.run(
                command,
                cwd=self.workspace,
                env=self._minimal_env(),
                timeout=self.timeout_seconds,
                preexec_fn=None if self.oci else self._resource_limits(),
            )
        except subprocess.TimeoutExpired:
            cleaned = not invocation or self.oci.cleanup(invocation.name, self._minimal_env())
            if not cleaned:
                log.error("oci_timeout_cleanup_failed", container=invocation.name)
            suffix = "" if cleaned else "; OCI cleanup failed"
            return {
                "success": False,
                "error": f"Timeout after {self.timeout_seconds}s{suffix}",
            }
        result = {
            "success": completed.returncode == 0,
            "result": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }
        if completed.stdout_truncated:
            result["stdout_truncated"] = True
        if completed.stderr_truncated:
            result["stderr_truncated"] = True
        return result

    def _exec_shell(self, command: str) -> dict[str, Any]:
        if not command.strip():
            return {"success": False, "error": "Empty command"}
        if len(command) > self.MAX_COMMAND_CHARS:
            return {"success": False, "error": "Command is too long"}
        if self.BLOCKED_SHELL_SYNTAX.search(command):
            return {"success": False, "error": "Shell operators and expansion are disabled"}
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            return {"success": False, "error": f"Invalid command syntax: {exc}"}
        if not argv or argv[0] not in self.allowed_commands:
            return {"success": False, "error": "Command is not allowlisted"}
        if Path(argv[0]).name != argv[0]:
            return {"success": False, "error": "Command paths are not allowed"}
        for argument in argv[1:]:
            if ".." in Path(argument).parts:
                return {"success": False, "error": "Parent path traversal is not allowed"}
            if argument.startswith("/") and not self._path_allowed(argument):
                return {"success": False, "error": f"Path is outside workspace: {argument}"}
            if argv[0] in {"ls", "head", "tail", "wc"} and not argument.startswith("-"):
                candidate = Path(argument)
                if not candidate.is_absolute():
                    candidate = self.workspace / candidate
                if not self._path_allowed(candidate):
                    return {"success": False, "error": f"Path is outside workspace: {argument}"}
            if argument.startswith("--output") or argument in {"-o", "--files0-from"}:
                return {"success": False, "error": "Output redirection options are disabled"}
        return self._run_process(argv)

    def _resolve_workspace_path(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        if not self._path_allowed(resolved):
            raise SecurityError(f"Path is outside workspace: {path}")
        return resolved

    def _exec_file_read(self, path: str) -> dict[str, Any]:
        target = self._resolve_workspace_path(path)
        if not target.is_file():
            return {"success": False, "error": f"File does not exist: {path}"}
        if target.stat().st_size > self.MAX_FILE_BYTES:
            return {"success": False, "error": "File exceeds size limit"}
        return {"success": True, "result": target.read_text(encoding="utf-8")[: self.MAX_OUTPUT]}

    def _exec_file_write(self, path: str, content: str) -> dict[str, Any]:
        encoded = content.encode("utf-8")
        if len(encoded) > self.MAX_FILE_BYTES:
            return {"success": False, "error": "Content exceeds size limit"}
        target = self._resolve_workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        return {"success": True, "result": f"Wrote {len(encoded)} bytes to {target}"}

    def _exec_web_request(self, url: str, method: str = "GET", body: Any = None) -> dict[str, Any]:
        self.network_policy.validate_url(url)
        method = method.upper()
        if method not in {"GET", "HEAD"} and not self.network_post_enabled:
            return {"success": False, "error": "Only GET and HEAD are enabled"}
        if method not in {"GET", "HEAD", "POST"}:
            return {"success": False, "error": f"Unsupported method: {method}"}
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=False) as client:
            response = client.request(method, url, content=body)
        if response.is_redirect:
            return {"success": False, "error": "Redirects are disabled"}
        return {
            "success": response.status_code < 400,
            "result": response.text[: self.MAX_OUTPUT],
            "status_code": response.status_code,
        }

    def _exec_python(self, code: str) -> dict[str, Any]:
        if len(code) > self.MAX_COMMAND_CHARS:
            return {"success": False, "error": "Python program is too long"}
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=self.workspace, encoding="utf-8"
        )
        try:
            with handle:
                handle.write(code)
            executable = "python3" if self.oci else sys.executable
            return self._run_process([executable, "-I", "-S", handle.name])
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def _path_allowed(self, path: str | Path) -> bool:
        try:
            Path(path).resolve().relative_to(self.workspace)
            return True
        except ValueError:
            return False
