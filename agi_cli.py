#!/usr/bin/env python3
"""CLI for scaffolding, signing, packing, validating, and inspecting `.agi` policies."""

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

import structlog

from runtime import __version__
from runtime.conformance import build_conformance_report
from runtime.loader import AGILoadError, load_agi, safe_extract_archive
from runtime.signing import (
    SignatureError,
    generate_keypair,
    load_private_key,
    load_public_key,
    sign_package,
)

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

sys.stdout.reconfigure(line_buffering=True)

_AGENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _template_dir() -> Path:
    candidates = [
        Path(os.environ["AGI_TEMPLATE_DIR"]) if os.environ.get("AGI_TEMPLATE_DIR") else None,
        Path(__file__).resolve().parent / "skeleton",
        Path(sys.prefix) / "share" / "agi-format" / "templates" / "minimal-agent",
    ]
    for candidate in candidates:
        if candidate and (candidate / "manifest.json").is_file():
            return candidate
    raise AGILoadError("Bundled minimal-agent template could not be located")


def cmd_init(args: argparse.Namespace) -> None:
    """Create a conforming package from the bundled minimal template."""
    target = Path(args.target)
    name = args.name or target.name
    if not _AGENT_NAME.fullmatch(name):
        print("Error: agent name must use 1-128 letters, digits, dots, underscores, or hyphens")
        sys.exit(1)
    if target.exists():
        print(f"Error: target already exists: {target}")
        sys.exit(1)

    try:
        shutil.copytree(_template_dir(), target)
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["agent"] = {
            "name": name,
            "version": args.version,
            "description": args.description or f"A conforming .agi agent package named {name}.",
            "license": args.license,
        }
        if args.author:
            manifest["agent"]["author"] = args.author
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with load_agi(target):
            pass
    except (AGILoadError, OSError, json.JSONDecodeError) as exc:
        shutil.rmtree(target, ignore_errors=True)
        print(f"Initialization failed: {exc}")
        sys.exit(1)
    print(f"Created conforming agent package: {target}")
    print(f"Next: agi validate {target}")


def cmd_pack(args: argparse.Namespace) -> None:
    """Pack a directory into a .agi archive."""
    source = Path(args.source)
    output = Path(args.output) if args.output else source.with_suffix(".agi")

    if not source.is_dir():
        print(f"Error: {source} is not a directory")
        sys.exit(1)
    try:
        output.resolve().relative_to(source.resolve())
    except ValueError:
        pass
    else:
        print("Error: output archive must be outside the source directory")
        sys.exit(1)

    # Validate first
    try:
        package = load_agi(source)
        print(f"Validated: {package.name} v{package.version}")
    except AGILoadError as e:
        print(f"Validation failed: {e}")
        sys.exit(1)

    # Pack deterministically and reject links/devices.
    print(f"Packing {source} -> {output}")
    entries = [source, *sorted(source.rglob("*"), key=lambda path: path.as_posix())]
    if any(path.is_symlink() for path in entries):
        print("Error: .agi packages may not contain symbolic links")
        sys.exit(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in entries:
                    arcname = Path(source.name) / path.relative_to(source)
                    info = tar.gettarinfo(str(path), arcname=str(arcname))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o755 if info.isdir() else 0o644
                    if info.isfile():
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
                    elif info.isdir():
                        tar.addfile(info)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Created {output} ({size_mb:.2f} MB)")


def cmd_unpack(args: argparse.Namespace) -> None:
    """Unpack a .agi archive to a directory."""
    source = Path(args.source)
    output = Path(args.output) if args.output else Path(source.stem)

    if not source.exists():
        print(f"Error: {source} does not exist")
        sys.exit(1)

    print(f"Unpacking {source} -> {output}")
    try:
        safe_extract_archive(source, output)
    except AGILoadError as exc:
        print(f"Extraction failed: {exc}")
        sys.exit(1)

    print(f"Extracted to {output}")


def _trusted_keys(paths: list[str] | None):
    if not paths:
        return None
    try:
        return [load_public_key(path) for path in paths]
    except SignatureError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a .agi directory or archive and optionally emit JSON evidence."""
    source_path = Path(args.source)
    if args.report:
        report_path = Path(args.report)
        if source_path.is_dir() and report_path.resolve().is_relative_to(source_path.resolve()):
            print("Error: conformance report must be written outside the package directory")
            sys.exit(1)
    report = build_conformance_report(source_path, trusted_keys=_trusted_keys(args.trusted_key))
    if args.report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["conformant"]:
        package = report["package"]
        signature = report["signature"]
        signed = f", signed by {signature['public_key_sha256'][:16]}" if signature else ", unsigned"
        print(f"Conformant: {package['name']} v{package['version']}{signed}")
        print(f"Content SHA-256: {package['content_sha256']}")
    else:
        print(f"Validation failed: {report['error']['message']}")
    if not report["conformant"]:
        sys.exit(1)


def cmd_inspect(args: argparse.Namespace) -> None:
    """Print the boundary a package declares."""
    try:
        package = load_agi(args.source)
    except AGILoadError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    with package:
        policy = package.policy
        sandbox = policy.sandbox
        print(f"{package.name} v{package.version} (.agi format {package.format_version})")
        if package.signature:
            print(f"  Signed by key {package.signature.public_key_sha256}")
        else:
            print("  Unsigned")
        print(f"  Content SHA-256: {package.digest.sha256}")
        print("\n  Tools:")
        for rule in policy.tools.values():
            approval = " [approval required]" if rule.requires_approval else ""
            print(f"    {rule.name} ({rule.scope}){approval}")
        if not policy.tools:
            print("    (none)")
        limit = policy.max_actions_per_session
        print(f"\n  Action budget per session: {'unlimited' if limit is None else limit}")
        network = sandbox["network"]
        domains = ", ".join(network["allowed_domains"]) or "none"
        print(f"  Network: domains={domains} http={network.get('allow_http', False)}"
              f" post={network.get('allow_post', False)}")
        execution = sandbox["execution"]
        print(f"  Execution: shell={execution['shell_enabled']} python={execution['python_enabled']}"
              f" commands={execution['allowed_commands']}")
        isolation = sandbox["isolation"]
        print(f"  Isolation: {isolation['type']} {isolation.get('image', '')}".rstrip())
        limits = sandbox["resource_limits"]
        print(f"  Limits: {limits['max_cpu_seconds_per_action']}s CPU, {limits['max_memory_mb']} MB")
        print("\n  Escalation triggers:")
        for trigger in policy.triggers:
            print(f"    [{trigger.severity}] {trigger.id}: {trigger.description}")
        if not policy.triggers:
            print("    (none)")


def cmd_check(args: argparse.Namespace) -> None:
    """Evaluate one proposed tool call against a package policy without executing it."""
    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print(f"Error: --params must be JSON: {exc}")
        sys.exit(1)
    try:
        with load_agi(args.source) as package:
            decision = package.policy.evaluate(args.tool, params)
    except AGILoadError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
    sys.exit(0 if decision.verdict.value == "allow" else 2)


def cmd_keygen(args: argparse.Namespace) -> None:
    """Create an Ed25519 signing key pair."""
    try:
        fingerprint = generate_keypair(args.private_key, args.public_key)
    except (SignatureError, OSError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Wrote {args.private_key} (keep secret) and {args.public_key}")
    print(f"Key fingerprint: {fingerprint}")


def cmd_sign(args: argparse.Namespace) -> None:
    """Sign a validated package directory in place."""
    source = Path(args.source)
    if not source.is_dir():
        print("Error: sign a package directory, then pack it")
        sys.exit(1)
    try:
        key = load_private_key(args.key)
        with load_agi(source):
            pass
        sign_package(source, key)
        with load_agi(source) as package:
            fingerprint = package.signature.public_key_sha256
    except (AGILoadError, SignatureError, OSError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Signed {source} with key {fingerprint}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=".agi CLI -- scaffold, sign, validate, pack, and inspect agent boundary policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Create a conforming starter package")
    p_init.add_argument("target", help="New package directory")
    p_init.add_argument("--name", help="Agent name (defaults to target directory name)")
    p_init.add_argument("--version", default="0.1.0", help="Initial agent version")
    p_init.add_argument("--author", help="Package author")
    p_init.add_argument("--description", help="Package description")
    p_init.add_argument("--license", default="MIT", help="SPDX license identifier")

    # pack
    p_pack = subparsers.add_parser("pack", help="Pack a directory into a .agi archive")
    p_pack.add_argument("source", help="Source directory")
    p_pack.add_argument("-o", "--output", help="Output .agi file path")

    # unpack
    p_unpack = subparsers.add_parser("unpack", help="Unpack a .agi archive")
    p_unpack.add_argument("source", help="Source .agi file")
    p_unpack.add_argument("-o", "--output", help="Output directory")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate a .agi directory or archive")
    p_validate.add_argument("source", help="Directory or .agi file to validate")
    p_validate.add_argument("--json", action="store_true", help="Print a machine-readable report")
    p_validate.add_argument("--report", help="Also write the JSON report to this path")
    p_validate.add_argument(
        "--trusted-key",
        action="append",
        help="PEM Ed25519 public key; require a valid signature by one of these (repeatable)",
    )

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a .agi file's contents")
    p_inspect.add_argument("source", help="Directory or .agi file to inspect")

    # check
    p_check = subparsers.add_parser("check", help="Evaluate a tool call against a policy")
    p_check.add_argument("source", help="Directory or .agi file")
    p_check.add_argument("tool", help="Tool name")
    p_check.add_argument("--params", default="{}", help="Tool parameters as a JSON object")

    # keygen
    p_keygen = subparsers.add_parser("keygen", help="Create an Ed25519 signing key pair")
    p_keygen.add_argument("--private-key", required=True, help="Private key output path")
    p_keygen.add_argument("--public-key", required=True, help="Public key output path")

    # sign
    p_sign = subparsers.add_parser("sign", help="Sign a package directory in place")
    p_sign.add_argument("source", help="Package directory")
    p_sign.add_argument("--key", required=True, help="PEM Ed25519 private key")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "pack": cmd_pack,
        "unpack": cmd_unpack,
        "validate": cmd_validate,
        "inspect": cmd_inspect,
        "check": cmd_check,
        "keygen": cmd_keygen,
        "sign": cmd_sign,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
