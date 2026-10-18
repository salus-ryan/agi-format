"""Tests for the .agi loader and CLI."""

import json
import subprocess
import sys
from pathlib import Path

from runtime.loader import load_agi

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ROOT / "skeleton"


def cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "agi_cli.py"), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def test_load_skeleton_exposes_policy():
    with load_agi(SKELETON) as package:
        assert package.name == "skeleton-agent"
        assert package.format_version == "0.2.0"
        assert set(package.policy.tools) == {"file_read", "file_write"}
        assert package.policy.tools["file_write"].requires_approval is True
        assert package.policy.max_actions_per_session == 100
        assert package.signature is None


def test_validate_and_inspect():
    assert cli("validate", str(SKELETON)).returncode == 0
    inspected = cli("inspect", str(SKELETON))
    assert inspected.returncode == 0
    assert "skeleton-agent" in inspected.stdout
    assert "file_write (fs:write) [approval required]" in inspected.stdout
    assert "credential_paths" in inspected.stdout


def test_check_reports_each_verdict():
    allowed = cli("check", str(SKELETON), "file_read", "--params", '{"path": "notes.txt"}')
    assert allowed.returncode == 0
    assert json.loads(allowed.stdout)["verdict"] == "allow"

    escalated = cli("check", str(SKELETON), "file_read", "--params", '{"path": ".env"}')
    assert escalated.returncode == 2
    assert json.loads(escalated.stdout)["trigger_id"] == "credential_paths"

    denied = cli("check", str(SKELETON), "shell", "--params", '{"command": "ls"}')
    assert denied.returncode == 2
    assert json.loads(denied.stdout)["verdict"] == "deny"


def test_pack_and_unpack(tmp_path):
    archive = tmp_path / "test.agi"
    assert cli("pack", str(SKELETON), "-o", str(archive)).returncode == 0
    unpacked = tmp_path / "unpacked"
    assert cli("unpack", str(archive), "-o", str(unpacked)).returncode == 0
    assert (unpacked / "skeleton" / "manifest.json").is_file()
    assert (unpacked / "skeleton" / "policy.json").is_file()
