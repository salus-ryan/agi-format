"""Conformance and archive-safety tests for the .agi format."""

import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from runtime.conformance import build_conformance_report
from runtime.loader import AGILoadError, load_agi, safe_extract_archive

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ROOT / "skeleton"
REPORT_SCHEMA = ROOT / "schemas/conformance-report-v0.2.0.json"


def copy_skeleton(tmp_path: Path) -> Path:
    target = tmp_path / "agent"
    shutil.copytree(SKELETON, target)
    return target


def edit_policy(package: Path, edit) -> None:
    path = package / "policy.json"
    policy = json.loads(path.read_text())
    edit(policy)
    path.write_text(json.dumps(policy))


def cli(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "agi_cli.py"), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


def test_schema_identifiers_are_immutable_and_skeleton_matches():
    for schema_path in (ROOT / "schemas").glob("*.json"):
        schema = json.loads(schema_path.read_text())
        assert "/v0.5.0/schemas/" in schema["$id"]
    for document, schema_name in (("manifest.json", "manifest"), ("policy.json", "policy")):
        data = json.loads((SKELETON / document).read_text())
        schema = json.loads((ROOT / f"schemas/{schema_name}-v0.2.0.json").read_text())
        assert data["$schema"] == schema["$id"]


def test_rejects_unsupported_format_version(tmp_path):
    package = copy_skeleton(tmp_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = "0.1.0"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AGILoadError, match="Unsupported format_version"):
        load_agi(package)


def test_rejects_missing_policy(tmp_path):
    package = copy_skeleton(tmp_path)
    (package / "policy.json").unlink()
    with pytest.raises(AGILoadError, match="Missing file: policy.json"):
        load_agi(package)


def test_rejects_unknown_policy_fields(tmp_path):
    package = copy_skeleton(tmp_path)
    edit_policy(package, lambda policy: policy.update({"alignment": {"no_harm": True}}))
    with pytest.raises(AGILoadError, match="schema violation"):
        load_agi(package)


def test_rejects_trigger_for_undeclared_tool(tmp_path):
    package = copy_skeleton(tmp_path)
    edit_policy(package, lambda policy: policy["escalation"]["triggers"][0].update(
        {"tools": ["web_request"]}
    ))
    with pytest.raises(AGILoadError, match="undeclared tools"):
        load_agi(package)


def test_rejects_process_tools_when_isolation_is_disabled(tmp_path):
    package = copy_skeleton(tmp_path)
    edit_policy(package, lambda policy: policy["sandbox"]["execution"].update(
        {"shell_enabled": True}
    ))
    with pytest.raises(AGILoadError, match="cannot enable shell"):
        load_agi(package)


def test_rejects_unpinned_oci_image(tmp_path):
    package = copy_skeleton(tmp_path)
    edit_policy(package, lambda policy: policy["sandbox"]["isolation"].update(
        {"type": "oci", "runtime": "docker", "image": "example/tool:latest"}
    ))
    with pytest.raises(AGILoadError, match="pinned"):
        load_agi(package)


def test_directory_rejects_symlinked_policy(tmp_path):
    package = copy_skeleton(tmp_path)
    policy = package / "policy.json"
    real = tmp_path / "real-policy.json"
    shutil.move(policy, real)
    try:
        policy.symlink_to(real)
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
        result = cli("pack", str(SKELETON), "-o", str(output))
        assert result.returncode == 0, result.stderr or result.stdout
    assert first.read_bytes() == second.read_bytes()


def test_conformance_report_matches_schema_and_archive_digest(tmp_path):
    archive = tmp_path / "agent.agi"
    result = cli("pack", str(SKELETON), "-o", str(archive))
    assert result.returncode == 0, result.stderr or result.stdout

    directory_report = build_conformance_report(SKELETON)
    archive_report = build_conformance_report(archive)
    schema = json.loads(REPORT_SCHEMA.read_text())
    Draft202012Validator(schema).validate(directory_report)
    Draft202012Validator(schema).validate(archive_report)
    assert directory_report["conformant"] is True
    assert directory_report["signature"] is None
    assert {"id": "signature", "status": "absent"} in directory_report["checks"]
    assert directory_report["package"]["content_sha256"] == archive_report["package"]["content_sha256"]


def test_report_cannot_modify_the_package_it_certifies():
    report_path = SKELETON / "conformance.json"
    result = cli("validate", str(SKELETON), "--report", str(report_path))
    assert result.returncode == 1
    assert "outside the package" in result.stdout
    assert not report_path.exists()


def test_invalid_package_produces_schema_valid_json_report(tmp_path):
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    result = cli("validate", str(invalid), "--json")
    assert result.returncode == 1
    report = json.loads(result.stdout)
    Draft202012Validator(json.loads(REPORT_SCHEMA.read_text())).validate(report)
    assert report["conformant"] is False
    assert report["error"]["code"] == "invalid-package"


def test_init_creates_custom_conforming_package_and_refuses_overwrite(tmp_path):
    target = tmp_path / "outside-agent"
    command = ("init", str(target), "--name", "outside-agent", "--author", "Independent Author")
    created = cli(*command, cwd=tmp_path)
    assert created.returncode == 0, created.stderr or created.stdout
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["agent"]["name"] == "outside-agent"
    assert manifest["agent"]["author"] == "Independent Author"
    assert build_conformance_report(target)["conformant"] is True

    repeated = cli(*command, cwd=tmp_path)
    assert repeated.returncode == 1
    assert "target already exists" in repeated.stdout
