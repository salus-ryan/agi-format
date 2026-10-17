"""Conformance and archive-safety tests for the .agi format."""

import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from runtime.loader import AGILoadError, load_agi, safe_extract_archive

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ROOT / "skeleton"


def copy_skeleton(tmp_path: Path) -> Path:
    target = tmp_path / "agent"
    shutil.copytree(SKELETON, target)
    return target


def test_rejects_unsupported_format_version(tmp_path):
    package = copy_skeleton(tmp_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = "9.0.0"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AGILoadError, match="schema violation|Unsupported"):
        load_agi(package)


def test_rejects_unknown_interface(tmp_path):
    package = copy_skeleton(tmp_path)
    module_path = package / "reasoning" / "module.json"
    module = json.loads(module_path.read_text())
    module["components"][0]["interface"] = "Unknown<Input, Output>"
    module_path.write_text(json.dumps(module))
    with pytest.raises(AGILoadError, match="unknown interface"):
        load_agi(package)


def test_rejects_component_path_traversal(tmp_path):
    package = copy_skeleton(tmp_path)
    module_path = package / "reasoning" / "module.json"
    module = json.loads(module_path.read_text())
    module["components"][0]["path"] = "../../manifest.json"
    module_path.write_text(json.dumps(module))
    with pytest.raises(AGILoadError, match="schema violation"):
        load_agi(package)


def test_directory_rejects_symlinked_component(tmp_path):
    package = copy_skeleton(tmp_path)
    component = package / "reasoning" / "inference" / "engine.json"
    component.unlink()
    try:
        component.symlink_to(package / "manifest.json")
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(AGILoadError, match="symlinks"):
        load_agi(package)


def test_archive_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "evil.agi"
    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"escape"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(AGILoadError, match="unsafe path"):
        safe_extract_archive(archive_path, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()


def test_archive_rejects_duplicate_paths(tmp_path):
    archive_path = tmp_path / "duplicate.agi"
    with tarfile.open(archive_path, "w:gz") as archive:
        for payload in (b"first", b"second"):
            info = tarfile.TarInfo("agent/manifest.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(AGILoadError, match="duplicate"):
        safe_extract_archive(archive_path, tmp_path / "output")


def test_pack_is_deterministic(tmp_path):
    first = tmp_path / "first.agi"
    second = tmp_path / "second.agi"
    for output in (first, second):
        result = subprocess.run(
            [sys.executable, str(ROOT / "agi_cli.py"), "pack", str(SKELETON), "-o", str(output)],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout
    assert first.read_bytes() == second.read_bytes()
