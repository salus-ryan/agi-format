"""Machine-readable conformance reports for `.agi` packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from runtime import __version__
from runtime.loader import AGILoadError, load_agi

REPORT_VERSION = "0.2.0"
REPORT_SCHEMA = (
    "https://raw.githubusercontent.com/salus-ryan/agi-format/"
    "v0.5.0/schemas/conformance-report-v0.2.0.json"
)


def build_conformance_report(
    source: str | Path, *, trusted_keys: list[Ed25519PublicKey] | None = None
) -> dict[str, Any]:
    """Validate a package and return a stable, JSON-serializable report."""
    source_path = Path(source)
    report: dict[str, Any] = {
        "$schema": REPORT_SCHEMA,
        "report_version": REPORT_VERSION,
        "implementation": {"name": "agi-format", "version": __version__},
        "source_kind": "directory" if source_path.is_dir() else "archive",
        "conformant": False,
        "format_version": None,
        "package": None,
        "signature": None,
        "checks": [],
        "error": None,
    }

    try:
        with load_agi(source_path, trusted_keys=trusted_keys) as package:
            signature = package.signature
            report.update(
                {
                    "conformant": True,
                    "format_version": package.format_version,
                    "package": {
                        "name": package.name,
                        "version": package.version,
                        "content_sha256": package.digest.sha256,
                        "file_count": package.digest.file_count,
                        "uncompressed_bytes": package.digest.total_bytes,
                    },
                    "signature": signature.to_dict() if signature else None,
                    "checks": [
                        {"id": "package-structure", "status": "pass"},
                        {"id": "json-schema", "status": "pass"},
                        {"id": "policy-semantics", "status": "pass"},
                        {"id": "signature", "status": "pass" if signature else "absent"},
                    ],
                }
            )
    except (AGILoadError, OSError) as exc:
        report["error"] = {"code": "invalid-package", "message": str(exc)}

    return report

