"""Machine-readable conformance reports for `.agi` packages."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from runtime import __version__
from runtime.loader import (
    FORMAT_VERSION,
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_MEMBERS,
    AGILoadError,
    load_agi,
)

REPORT_VERSION = "0.1.0"
REPORT_SCHEMA = (
    "https://raw.githubusercontent.com/salus-ryan/agi-format/"
    "v0.3.0/schemas/conformance-report-v0.1.0.json"
)


def _content_digest(root: Path) -> tuple[str, int, int]:
    """Hash the complete package tree without following links or special files."""
    files: list[Path] = []
    total_bytes = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise AGILoadError(f"Package may not contain symbolic links: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AGILoadError(f"Package contains unsupported entry: {relative}")
        files.append(path)
        total_bytes += path.stat().st_size
        if len(files) > MAX_ARCHIVE_MEMBERS:
            raise AGILoadError(f"Package has too many files ({len(files)})")
        if total_bytes > MAX_ARCHIVE_BYTES:
            raise AGILoadError(f"Package exceeds {MAX_ARCHIVE_BYTES} bytes")

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files), total_bytes


def build_conformance_report(source: str | Path) -> dict[str, Any]:
    """Validate a package and return a stable, JSON-serializable report."""
    source_path = Path(source)
    source_kind = "directory" if source_path.is_dir() else "archive"
    base: dict[str, Any] = {
        "$schema": REPORT_SCHEMA,
        "report_version": REPORT_VERSION,
        "implementation": {"name": "agi-format", "version": __version__},
        "source_kind": source_kind,
        "conformant": False,
        "format_version": None,
        "package": None,
        "checks": [],
        "error": None,
    }

    try:
        with load_agi(source_path) as package:
            digest, file_count, total_bytes = _content_digest(package.root)
            base.update(
                {
                    "conformant": True,
                    "format_version": package.format_version,
                    "package": {
                        "name": package.name,
                        "version": package.version,
                        "content_sha256": digest,
                        "file_count": file_count,
                        "uncompressed_bytes": total_bytes,
                    },
                    "checks": [
                        {"id": "package-structure", "status": "pass"},
                        {"id": "json-schema", "status": "pass"},
                        {"id": "interface-registry", "status": "pass"},
                        {"id": "safety-profile", "status": "pass"},
                    ],
                }
            )
    except (AGILoadError, OSError) as exc:
        base["error"] = {"code": "invalid-package", "message": str(exc)}

    return base


def assert_report_shape(report: dict[str, Any]) -> None:
    """Defensively ensure report metadata matches this implementation."""
    if report.get("report_version") != REPORT_VERSION:
        raise ValueError("Unexpected conformance report version")
    if report.get("format_version") not in {None, FORMAT_VERSION}:
        raise ValueError("Unexpected format version in conformance report")
