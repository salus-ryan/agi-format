"""Tests for the .agi loader and CLI."""

import subprocess
import sys

from runtime.loader import load_agi

SKELETON_PATH = "skeleton"


class TestLoader:
    def test_load_skeleton(self):
        pkg = load_agi(SKELETON_PATH)
        assert pkg.name == "skeleton-agent"
        assert pkg.version == "0.1.0"

    def test_load_json(self):
        pkg = load_agi(SKELETON_PATH)
        alignment = pkg.load_json("goals/alignment/alignment.json")
        assert "config" in alignment
        assert "constraints" in alignment["config"]

    def test_required_safety_files(self):
        pkg = load_agi(SKELETON_PATH)
        # These must exist for a valid .agi package
        alignment = pkg.load_json("goals/alignment/alignment.json")
        sandbox = pkg.load_json("action/sandbox/sandbox.json")
        escalation = pkg.load_json("goals/escalation/escalation.json")
        assert alignment is not None
        assert sandbox is not None
        assert escalation is not None


class TestCLI:
    def test_validate(self):
        result = subprocess.run(
            [sys.executable, "agi_cli.py", "validate", SKELETON_PATH],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_inspect(self):
        result = subprocess.run(
            [sys.executable, "agi_cli.py", "inspect", SKELETON_PATH],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "skeleton-agent" in result.stdout

    def test_pack_and_unpack(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            agi_path = os.path.join(tmpdir, "test.agi")
            # Pack
            result = subprocess.run(
                [sys.executable, "agi_cli.py", "pack", SKELETON_PATH, "-o", agi_path],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            assert os.path.exists(agi_path)

            # Unpack
            unpack_dir = os.path.join(tmpdir, "unpacked")
            result = subprocess.run(
                [sys.executable, "agi_cli.py", "unpack", agi_path, "-o", unpack_dir],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            # Unpacked archive has skeleton/ subdir containing manifest.json
            assert os.path.exists(os.path.join(unpack_dir, "skeleton", "manifest.json"))
