#!/usr/bin/env python3
"""CLI tool for .agi files -- pack, unpack, validate, inspect, and run.

Usage:
    python agi_cli.py pack skeleton/ -o agent.agi
    python agi_cli.py unpack agent.agi -o agent_unpacked/
    python agi_cli.py validate skeleton/
    python agi_cli.py inspect agent.agi
    python agi_cli.py run skeleton/
    python agi_cli.py run agent.agi
"""

import argparse
import json
import sys
import tarfile
from pathlib import Path

import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)

sys.stdout.reconfigure(line_buffering=True)

from runtime.loader import AGILoadError, load_agi


def cmd_pack(args: argparse.Namespace) -> None:
    """Pack a directory into a .agi archive."""
    source = Path(args.source)
    output = Path(args.output) if args.output else source.with_suffix(".agi")

    if not source.is_dir():
        print(f"Error: {source} is not a directory")
        sys.exit(1)

    # Validate first
    try:
        package = load_agi(source)
        print(f"Validated: {package.name} v{package.version}")
    except AGILoadError as e:
        print(f"Validation failed: {e}")
        sys.exit(1)

    # Pack
    print(f"Packing {source} -> {output}")
    with tarfile.open(output, "w:gz") as tar:
        tar.add(source, arcname=source.name)

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
    with tarfile.open(source, "r:gz") as tar:
        tar.extractall(output, filter="data")

    print(f"Extracted to {output}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a .agi directory or archive."""
    source = Path(args.source)
    try:
        package = load_agi(source)
        print(f"VALID: {package.name} v{package.version} (format v{package.format_version})")

        # Deep validation: load all modules and components
        errors = []
        for module_name in ["perception", "reasoning", "memory", "language", "action", "learning", "goals"]:
            try:
                module = package.load_module(module_name)
                for comp in module.get("components", []):
                    try:
                        package.load_json(f"{module_name}/{comp['path']}")
                    except Exception as e:
                        errors.append(f"  {module_name}/{comp['path']}: {e}")
            except Exception as e:
                errors.append(f"  {module_name}: {e}")

        if errors:
            print(f"\nWarnings ({len(errors)}):")
            for err in errors:
                print(err)
        else:
            print("All modules and components validated successfully.")

    except AGILoadError as e:
        print(f"INVALID: {e}")
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
    print(f"  .agi Package Inspector")
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
    print(f"\n  Runtime Requirements:")
    print(f"    Min memory:      {runtime.get('min_memory_gb', '?')} GB")
    print(f"    GPU required:    {runtime.get('gpu_required', False)}")
    print(f"    GPU VRAM:        {runtime.get('min_gpu_vram_gb', 'N/A')} GB")
    print(f"    Sandbox:         {'required' if runtime.get('sandbox_required') else 'optional'}")
    print(f"    Entrypoint:      {runtime.get('entrypoint')}")

    print(f"\n  Capabilities:")
    for module_name in ["perception", "reasoning", "memory", "language", "action", "learning", "goals"]:
        module = package.load_module(module_name)
        components = module.get("components", [])
        deps = module.get("dependencies", [])
        print(f"    {module_name}:")
        print(f"      Components: {', '.join(c['name'] for c in components)}")
        if deps:
            print(f"      Depends on: {', '.join(deps)}")

    # Safety inspection
    print(f"\n  Safety Configuration:")
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
    print(f"\n  Agent Loop:")
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
        description=".agi file format CLI -- pack, unpack, validate, inspect, and run AGI agent blueprints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a .agi file's contents")
    p_inspect.add_argument("source", help="Directory or .agi file to inspect")

    # run
    p_run = subparsers.add_parser("run", help="Run a .agi agent interactively")
    p_run.add_argument("source", help="Directory or .agi file to run")
    p_run.add_argument("--max-steps", type=int, default=100, help="Max interaction steps")

    args = parser.parse_args()

    commands = {
        "pack": cmd_pack,
        "unpack": cmd_unpack,
        "validate": cmd_validate,
        "inspect": cmd_inspect,
        "run": cmd_run,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
