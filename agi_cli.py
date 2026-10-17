#!/usr/bin/env python3
"""CLI for scaffolding, packing, validating, inspecting, and running `.agi` files."""

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


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a .agi directory or archive and optionally emit JSON evidence."""
    source_path = Path(args.source)
    if args.report:
        report_path = Path(args.report)
        if source_path.is_dir() and report_path.resolve().is_relative_to(source_path.resolve()):
            print("Error: conformance report must be written outside the package directory")
            sys.exit(1)
    report = build_conformance_report(source_path)
    if args.report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["conformant"]:
        package = report["package"]
        print(
            f"VALID: {package['name']} v{package['version']} "
            f"(format v{report['format_version']})"
        )
        print("All modules and components validated successfully.")
        print(f"Content SHA-256: {package['content_sha256']}")
    else:
        print(f"INVALID: {report['error']['message']}")
    if not report["conformant"]:
        sys.exit(1)


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect the contents of a .agi file."""
    source = Path(args.source)
    try:
        package = load_agi(source)
    except AGILoadError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  .agi Package Inspector")
    print(f"{'='*60}")
    print(f"\n  Agent:   {package.name}")
    print(f"  Version: {package.version}")
    print(f"  Format:  v{package.format_version}")

    agent = package.manifest["agent"]
    if agent.get("description"):
        print(f"  Desc:    {agent['description']}")
    if agent.get("author"):
        print(f"  Author:  {agent['author']}")
    if agent.get("license"):
        print(f"  License: {agent['license']}")

    runtime = package.manifest["runtime"]
    print("\n  Runtime Requirements:")
    print(f"    Min memory:      {runtime.get('min_memory_gb', '?')} GB")
    print(f"    GPU required:    {runtime.get('gpu_required', False)}")
    print(f"    GPU VRAM:        {runtime.get('min_gpu_vram_gb', 'N/A')} GB")
    print(f"    Sandbox:         {'required' if runtime.get('sandbox_required') else 'optional'}")
    print(f"    Entrypoint:      {runtime.get('entrypoint')}")

    print("\n  Capabilities:")
    for module_name in ["perception", "reasoning", "memory", "language", "action", "learning", "goals"]:
        module = package.load_module(module_name)
        components = module.get("components", [])
        deps = module.get("dependencies", [])
        print(f"    {module_name}:")
        print(f"      Components: {', '.join(c['name'] for c in components)}")
        if deps:
            print(f"      Depends on: {', '.join(deps)}")

    # Safety inspection
    print("\n  Safety Configuration:")
    alignment = package.load_json("goals/alignment/alignment.json")
    constraints = alignment["config"]["constraints"]
    hard = [c for c in constraints if c["type"] == "hard"]
    soft = [c for c in constraints if c["type"] == "soft"]
    print(f"    Alignment constraints: {len(hard)} hard, {len(soft)} soft")
    for c in hard:
        print(f"      [HARD] {c['id']}: {c['description']}")
    for c in soft:
        print(f"      [soft] {c['id']}: {c['description']}")

    escalation = package.load_json("goals/escalation/escalation.json")
    triggers = escalation["config"]["escalation_triggers"]
    print(f"    Escalation triggers: {len(triggers)}")
    for t in triggers:
        print(f"      [{t['severity']}] {t['id']}: {t['description']}")

    # Goal loop
    print("\n  Agent Loop:")
    goal_loop = package.load_goal_loop()
    for step in goal_loop.get("steps", []):
        if "loop" in step and step.get("over"):
            print(f"    {step['name']} (loop over {step['over']}):")
            for substep in step.get("steps", []):
                desc = substep.get("description", "")
                print(f"      -> {substep['name']}: {desc}")
        else:
            desc = step.get("description", "")
            cond = f" [if {step['condition']}]" if step.get("condition") else ""
            print(f"    -> {step['name']}: {desc}{cond}")

    print(f"\n{'='*60}\n")


def cmd_run(args: argparse.Namespace) -> None:
    """Load and run a .agi agent interactively."""
    source = Path(args.source)
    try:
        package = load_agi(source)
    except AGILoadError as e:
        print(f"Error: {e}")
        sys.exit(1)

    from runtime.kernel import AGIKernel

    kernel = AGIKernel(package)
    kernel.run_interactive(max_steps=args.max_steps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=".agi format CLI -- scaffold, validate, pack, inspect, and run agent blueprints",
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

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a .agi file's contents")
    p_inspect.add_argument("source", help="Directory or .agi file to inspect")

    # run
    p_run = subparsers.add_parser("run", help="Run a .agi agent interactively")
    p_run.add_argument("source", help="Directory or .agi file to run")
    p_run.add_argument("--max-steps", type=int, default=100, help="Max interaction steps")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "pack": cmd_pack,
        "unpack": cmd_unpack,
        "validate": cmd_validate,
        "inspect": cmd_inspect,
        "run": cmd_run,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
