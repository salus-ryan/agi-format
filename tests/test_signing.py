"""Package signing and trust tests."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from runtime.conformance import build_conformance_report
from runtime.loader import AGILoadError, load_agi
from runtime.signing import generate_keypair, load_private_key, load_public_key, sign_package

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ROOT / "skeleton"


@pytest.fixture
def keys(tmp_path):
    private, public = tmp_path / "signing.pem", tmp_path / "signing.pub"
    generate_keypair(private, public)
    return private, public


@pytest.fixture
def package(tmp_path):
    target = tmp_path / "agent"
    shutil.copytree(SKELETON, target)
    return target


def test_keygen_refuses_overwrite_and_protects_private_key(keys):
    private, public = keys
    assert private.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="overwrite"):
        generate_keypair(private, public)


def test_signed_package_verifies_against_trusted_key(package, keys):
    private, public = keys
    before = load_agi(package).digest.sha256
    sign_package(package, load_private_key(private))
    loaded = load_agi(package, trusted_keys=[load_public_key(public)])
    assert loaded.digest.sha256 == before
    assert loaded.signature.trusted is True
    assert load_agi(package).signature.trusted is None


def test_tampering_after_signing_is_rejected(package, keys):
    private, _ = keys
    sign_package(package, load_private_key(private))
    policy_path = package / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["tools"][1]["requires_approval"] = False
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(AGILoadError, match="do not match the signed digest"):
        load_agi(package)


def test_forged_signature_is_rejected(package, keys, tmp_path):
    private, _ = keys
    sign_package(package, load_private_key(private))
    other_private, _ = tmp_path / "other.pem", tmp_path / "other.pub"
    generate_keypair(other_private, tmp_path / "other.pub")
    signature_path = package / "signature.json"
    document = json.loads(signature_path.read_text())
    forged = sign_package(package, load_private_key(other_private))
    document["signature"] = forged["signature"]
    signature_path.write_text(json.dumps(document))
    with pytest.raises(AGILoadError, match="signature is invalid"):
        load_agi(package)


def test_untrusted_and_unsigned_packages_fail_when_trust_is_required(package, keys, tmp_path):
    _, public = keys
    with pytest.raises(AGILoadError, match="unsigned"):
        load_agi(package, trusted_keys=[load_public_key(public)])
    stranger_private = tmp_path / "stranger.pem"
    generate_keypair(stranger_private, tmp_path / "stranger.pub")
    sign_package(package, load_private_key(stranger_private))
    with pytest.raises(AGILoadError, match="untrusted key"):
        load_agi(package, trusted_keys=[load_public_key(public)])


def test_cli_sign_pack_and_validate_archive_with_trusted_key(package, keys, tmp_path):
    private, public = keys

    def cli(*args):
        return subprocess.run(
            [sys.executable, str(ROOT / "agi_cli.py"), *args],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )

    signed = cli("sign", str(package), "--key", str(private))
    assert signed.returncode == 0, signed.stdout
    archive = tmp_path / "agent.agi"
    assert cli("pack", str(package), "-o", str(archive)).returncode == 0
    validated = cli("validate", str(archive), "--json", "--trusted-key", str(public))
    assert validated.returncode == 0, validated.stdout
    report = json.loads(validated.stdout)
    schema = json.loads((ROOT / "schemas/conformance-report-v0.2.0.json").read_text())
    Draft202012Validator(schema).validate(report)
    assert report["signature"]["trusted"] is True
    assert {"id": "signature", "status": "pass"} in report["checks"]
    assert build_conformance_report(archive)["signature"]["trusted"] is None
