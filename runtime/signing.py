"""Content digests and Ed25519 signatures for `.agi` packages."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

SIGNATURE_FILE = "signature.json"
SIGNATURE_DOMAIN = b"agi-format package signature v0.2.0\n"
SIGNATURE_SCHEMA = (
    "https://raw.githubusercontent.com/salus-ryan/agi-format/"
    "v0.5.0/schemas/signature-v0.2.0.json"
)
MAX_PACKAGE_FILES = 10_000
MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024


class SignatureError(ValueError):
    """Raised when a package digest or signature cannot be produced or verified."""


@dataclass(frozen=True)
class Digest:
    sha256: str
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class SignatureStatus:
    """A verified package signature. `trusted` is None when no trust set was supplied."""

    public_key_sha256: str
    trusted: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": "ed25519",
            "public_key_sha256": self.public_key_sha256,
            "trusted": self.trusted,
        }


def content_digest(root: str | Path) -> Digest:
    """Hash every package file except the root signature, in sorted path order."""
    root = Path(root)
    files: list[tuple[str, Path]] = []
    total = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise SignatureError(f"Package may not contain symbolic links: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise SignatureError(f"Package contains unsupported entry: {relative}")
        if relative == SIGNATURE_FILE:
            continue
        files.append((relative, path))
        total += path.stat().st_size
        if len(files) > MAX_PACKAGE_FILES:
            raise SignatureError(f"Package has too many files ({len(files)})")
        if total > MAX_PACKAGE_BYTES:
            raise SignatureError(f"Package exceeds {MAX_PACKAGE_BYTES} bytes")

    digest = hashlib.sha256()
    for relative, path in sorted(files):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return Digest(digest.hexdigest(), len(files), total)


def _raw_public_key(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def public_key_fingerprint(key: Ed25519PublicKey) -> str:
    return hashlib.sha256(_raw_public_key(key)).hexdigest()


def generate_keypair(private_path: str | Path, public_path: str | Path) -> str:
    """Write a new PEM key pair, refusing to overwrite. Returns the key fingerprint."""
    private_path, public_path = Path(private_path), Path(public_path)
    for path in (private_path, public_path):
        if path.exists():
            raise SignatureError(f"Refusing to overwrite existing file: {path}")
    key = Ed25519PrivateKey.generate()
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(private_pem)
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return public_key_fingerprint(key.public_key())


def load_private_key(path: str | Path) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise SignatureError(f"Cannot load private key {path}: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise SignatureError("Signing key must be an Ed25519 private key")
    return key


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise SignatureError(f"Cannot load public key {path}: {exc}") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise SignatureError("Trusted key must be an Ed25519 public key")
    return key


def sign_package(root: str | Path, private_key: Ed25519PrivateKey) -> dict[str, str]:
    """Write `signature.json` at the package root over its current content digest."""
    root = Path(root)
    digest = content_digest(root)
    signature = private_key.sign(SIGNATURE_DOMAIN + digest.sha256.encode("ascii"))
    document = {
        "$schema": SIGNATURE_SCHEMA,
        "algorithm": "ed25519",
        "public_key": base64.b64encode(_raw_public_key(private_key.public_key())).decode(),
        "content_sha256": digest.sha256,
        "signature": base64.b64encode(signature).decode(),
    }
    (root / SIGNATURE_FILE).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def verify_signature(
    document: dict[str, str],
    digest: Digest,
    trusted_keys: list[Ed25519PublicKey] | None = None,
) -> SignatureStatus:
    """Verify a schema-valid signature document against a freshly computed digest."""
    if document["content_sha256"] != digest.sha256:
        raise SignatureError("Package contents do not match the signed digest")
    try:
        raw_key = base64.b64decode(document["public_key"], validate=True)
        signature = base64.b64decode(document["signature"], validate=True)
        public_key = Ed25519PublicKey.from_public_bytes(raw_key)
        public_key.verify(signature, SIGNATURE_DOMAIN + digest.sha256.encode("ascii"))
    except (ValueError, InvalidSignature) as exc:
        raise SignatureError("Package signature is invalid") from exc

    fingerprint = public_key_fingerprint(public_key)
    trusted = None
    if trusted_keys is not None:
        trusted = fingerprint in {public_key_fingerprint(key) for key in trusted_keys}
        if not trusted:
            raise SignatureError(f"Package is signed by an untrusted key ({fingerprint[:16]})")
    return SignatureStatus(fingerprint, trusted)
