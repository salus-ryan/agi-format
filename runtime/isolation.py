"""Bounded process execution and an opt-in OCI isolation boundary."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from runtime.security import SecurityError

_IMAGE_BY_DIGEST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}$")
_OCI_RUNTIMES = frozenset({"docker", "podman"})


@dataclass(frozen=True)
class OCIInvocation:
    """One named OCI execution that can be force-removed after client failure."""

    argv: list[str]
    name: str


@dataclass(frozen=True)
class ProcessOutcome:
    """Bounded process result."""

    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool


class BoundedProcessRunner:
    """Drain process pipes continuously while retaining only bounded output."""

    def __init__(self, max_output_bytes: int = 10_000):
        self.max_output_bytes = max(1, int(max_output_bytes))

    def run(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
        preexec_fn: Callable[[], None] | None = None,
    ) -> ProcessOutcome:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=os.name == "posix",
            preexec_fn=preexec_fn,
        )
        stdout = bytearray()
        stderr = bytearray()
        truncated = [False, False]

        def drain(stream, destination: bytearray, index: int) -> None:
            while chunk := stream.read(8192):
                remaining = self.max_output_bytes - len(destination)
                if remaining > 0:
                    destination.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated[index] = True

        assert process.stdout is not None and process.stderr is not None
        threads = [
            threading.Thread(target=drain, args=(process.stdout, stdout, 0), daemon=True),
            threading.Thread(target=drain, args=(process.stderr, stderr, 1), daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._terminate(process)
            for thread in threads:
                thread.join(timeout=2)
            raise
        for thread in threads:
            thread.join(timeout=2)
        return ProcessOutcome(
            returncode=returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            stdout_truncated=truncated[0],
            stderr_truncated=truncated[1],
        )

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


class OCIIsolation:
    """Build a rootless-compatible, networkless OCI invocation."""

    def __init__(self, runtime: str, image: str, workspace: Path, max_memory_mb: int):
        if os.name != "posix":
            raise SecurityError("OCI process isolation requires a POSIX host")
        if runtime not in _OCI_RUNTIMES:
            raise SecurityError("OCI runtime must be docker or podman")
        executable_dir = str(Path(sys.executable).resolve().parent)
        safe_path = os.pathsep.join(dict.fromkeys([executable_dir, *os.defpath.split(os.pathsep)]))
        executable = shutil.which(runtime, path=safe_path)
        if executable is None:
            raise SecurityError(f"OCI runtime is unavailable: {runtime}")
        if not _IMAGE_BY_DIGEST.fullmatch(image):
            raise SecurityError("OCI image must be pinned by sha256 digest")
        self.executable = executable
        self.image = image
        self.workspace = workspace.resolve()
        self.max_memory_mb = max_memory_mb

    def command(self, argv: list[str]) -> OCIInvocation:
        mapped = [self._map_argument(argument) for argument in argv]
        name = f"agi-tool-{uuid.uuid4().hex}"
        command = [
            self.executable,
            "run",
            "--rm",
            "--name",
            name,
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=16",
            f"--memory={self.max_memory_mb}m",
            "--cpus=1",
            f"--user={os.getuid()}:{os.getgid()}",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
            f"--mount=type=bind,src={self.workspace},dst=/workspace,rw",
            "--workdir=/workspace",
            self.image,
            *mapped,
        ]
        return OCIInvocation(command, name)

    def cleanup(self, name: str, env: dict[str, str]) -> bool:
        """Force-remove a container whose attached client timed out."""
        try:
            completed = subprocess.run(
                [self.executable, "rm", "--force", name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                shell=False,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0

    def _map_argument(self, argument: str) -> str:
        path = Path(argument)
        if not path.is_absolute():
            return argument
        try:
            relative = path.resolve().relative_to(self.workspace)
        except ValueError as exc:
            raise SecurityError(f"OCI argument path is outside workspace: {argument}") from exc
        return str(Path("/workspace") / relative)
