"""Secure loader and validator for `.agi` boundary-policy packages."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import structlog
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from jsonschema import Draft202012Validator, FormatChecker

from runtime.policy import Policy, PolicyError
from runtime.signing import (
    MAX_PACKAGE_BYTES,
    MAX_PACKAGE_FILES,
    SIGNATURE_FILE,
    Digest,
    SignatureError,
    SignatureStatus,
    content_digest,
    verify_signature,
)

log = structlog.get_logger()

FORMAT_VERSION = "0.2.0"
SUPPORTED_FORMAT_VERSIONS = {FORMAT_VERSION}
MANIFEST_FILE = "manifest.json"
POLICY_FILE = "policy.json"
MAX_ARCHIVE_MEMBERS = MAX_PACKAGE_FILES
MAX_ARCHIVE_BYTES = MAX_PACKAGE_BYTES


class AGILoadError(Exception):
    """Raised when a .agi package is malformed, unsafe, or incompatible."""


def _schema_dir() -> Path:
    candidates = [
        Path(os.environ["AGI_SCHEMA_DIR"]) if os.environ.get("AGI_SCHEMA_DIR") else None,
        Path(__file__).resolve().parent.parent / "schemas",
        Path(sys.prefix) / "share" / "agi-format" / "schemas",
    ]
    for candidate in candidates:
        if candidate and candidate.is_dir():
            return candidate
    raise AGILoadError("Bundled .agi JSON Schemas could not be located")


def _load_schema(name: str) -> dict[str, Any]:
    path = _schema_dir() / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AGILoadError(f"Cannot load schema {name}: {exc}") from exc


def _validate_schema(instance: Any, schema_name: str, label: str) -> None:
    validator = Draft202012Validator(
        _load_schema(schema_name), format_checker=FormatChecker()
    )
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise AGILoadError(f"{label}: schema violation at {location}: {error.message}")


def _safe_relative_path(value: str, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise AGILoadError(f"{label}: path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise AGILoadError(f"{label}: unsafe path {value!r}")
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_extract_archive(source: str | Path, destination: str | Path) -> Path:
    """Extract a .agi tarball without links, traversal, devices, or archive bombs."""
    source = Path(source)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    try:
        archive = tarfile.open(source, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise AGILoadError(f"Cannot open archive: {exc}") from exc

    with archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise AGILoadError(f"Archive has too many entries ({len(members)})")
        total_size = sum(member.size for member in members if member.isfile())
        if total_size > MAX_ARCHIVE_BYTES:
            raise AGILoadError(f"Archive expands beyond {MAX_ARCHIVE_BYTES} bytes")
        normalized_names = [str(PurePosixPath(member.name)) for member in members]
        if len(normalized_names) != len(set(normalized_names)):
            raise AGILoadError("Archive contains duplicate paths")

        for member in members:
            relative = _safe_relative_path(member.name, "archive member")
            target = (destination / Path(*relative.parts)).resolve()
            if not _is_within(target, destination):
                raise AGILoadError(f"Archive member escapes destination: {member.name}")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise AGILoadError(f"Archive contains unsupported entry: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise AGILoadError(f"Archive contains unsupported entry: {member.name}")

        for member in members:
            relative = PurePosixPath(member.name)
            target = destination / Path(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise AGILoadError(f"Cannot extract archive member: {member.name}")
            with extracted, target.open("wb") as output:
                shutil.copyfileobj(extracted, output)

    children = [child for child in destination.iterdir() if child.name != "__MACOSX"]
    return children[0] if len(children) == 1 and children[0].is_dir() else destination


class AGIPackage:
    """A validated `.agi` package with path-confined access to its files."""

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        policy: Policy,
        digest: Digest,
        signature: SignatureStatus | None,
        temporary_directory: tempfile.TemporaryDirectory[str] | None = None,
    ):
        self.root = root.resolve()
        self.manifest = manifest
        self.policy = policy
        self.digest = digest
        self.signature = signature
        self._temporary_directory = temporary_directory

    @property
    def name(self) -> str:
        return self.manifest["agent"]["name"]

    @property
    def version(self) -> str:
        return self.manifest["agent"]["version"]

    @property
    def format_version(self) -> str:
        return self.manifest["format_version"]

    def resolve_path(self, relative_path: str) -> Path:
        return _resolve_confined(self.root, relative_path)

    def load_json(self, relative_path: str) -> dict[str, Any]:
        """Load an object-valued JSON file confined to the package root."""
        return _load_json(self.root, relative_path)

    def close(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> AGIPackage:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _resolve_confined(root: Path, relative_path: str) -> Path:
    relative = _safe_relative_path(relative_path, "package reference")
    candidate = root / Path(*relative.parts)
    current = candidate
    while current != root:
        if current.is_symlink():
            raise AGILoadError(f"Package reference may not use symlinks: {relative_path}")
        current = current.parent
    full_path = candidate.resolve()
    if not _is_within(full_path, root):
        raise AGILoadError(f"Package reference escapes root: {relative_path}")
    return full_path


def _load_json(root: Path, relative_path: str) -> dict[str, Any]:
    full_path = _resolve_confined(root, relative_path)
    if not full_path.is_file():
        raise AGILoadError(f"Missing file: {relative_path}")
    try:
        value = json.loads(full_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AGILoadError(f"Invalid JSON in {relative_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AGILoadError(f"Expected a JSON object in {relative_path}")
    return value


def _validate(
    root: Path, trusted_keys: list[Ed25519PublicKey] | None
) -> tuple[dict[str, Any], Policy, Digest, SignatureStatus | None]:
    manifest = _load_json(root, MANIFEST_FILE)
    version = manifest.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_FORMAT_VERSIONS))
        raise AGILoadError(f"Unsupported format_version {version!r}; supported: {supported}")
    _validate_schema(manifest, "manifest-v0.2.0.json", MANIFEST_FILE)

    policy_data = _load_json(root, POLICY_FILE)
    _validate_schema(policy_data, "policy-v0.2.0.json", POLICY_FILE)
    try:
        policy = Policy.from_dict(policy_data)
    except PolicyError as exc:
        raise AGILoadError(f"{POLICY_FILE}: {exc}") from exc

    try:
        digest = content_digest(root)
    except SignatureError as exc:
        raise AGILoadError(str(exc)) from exc

    signature: SignatureStatus | None = None
    signature_path = root / SIGNATURE_FILE
    if signature_path.exists() or signature_path.is_symlink():
        document = _load_json(root, SIGNATURE_FILE)
        _validate_schema(document, "signature-v0.2.0.json", SIGNATURE_FILE)
        try:
            signature = verify_signature(document, digest, trusted_keys)
        except SignatureError as exc:
            raise AGILoadError(str(exc)) from exc
    elif trusted_keys is not None:
        raise AGILoadError("Package is unsigned but a trusted signing key is required")
    return manifest, policy, digest, signature


def load_agi(
    path: str | Path, *, trusted_keys: list[Ed25519PublicKey] | None = None
) -> AGIPackage:
    """Load and fully validate a `.agi` directory or archive.

    When `trusted_keys` is given, the package must carry a valid signature by one of them.
    A present but invalid signature is always rejected.
    """
    path = Path(path)
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None

    if path.is_dir():
        root = path.resolve()
        log.info("loading_agi_directory", path=str(path))
    elif path.is_file() and (path.suffix == ".agi" or path.suffixes[-2:] == [".tar", ".gz"]):
        temporary_directory = tempfile.TemporaryDirectory(prefix="agi_")
        log.info("extracting_agi_archive", path=str(path), to=temporary_directory.name)
        try:
            root = safe_extract_archive(path, temporary_directory.name)
        except Exception:
            temporary_directory.cleanup()
            raise
    else:
        raise AGILoadError(f"Unsupported or missing package path: {path}")

    try:
        if not (root / MANIFEST_FILE).is_file():
            raise AGILoadError("Missing manifest.json in .agi package")
        manifest, policy, digest, signature = _validate(root, trusted_keys)
    except Exception:
        if temporary_directory:
            temporary_directory.cleanup()
        raise

    package = AGIPackage(root, manifest, policy, digest, signature, temporary_directory)
    log.info(
        "agi_loaded",
        name=package.name,
        version=package.version,
        format_version=package.format_version,
        signed=signature is not None,
    )
    return package
