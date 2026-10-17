"""Secure loader and conformance validator for .agi packages."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import structlog
from jsonschema import Draft202012Validator, FormatChecker

log = structlog.get_logger()

FORMAT_VERSION = "0.1.0"
SUPPORTED_FORMAT_VERSIONS = {FORMAT_VERSION}
REQUIRED_MODULES = [
    "perception",
    "reasoning",
    "memory",
    "language",
    "action",
    "learning",
    "goals",
]
REQUIRED_SAFETY_FILES = [
    "goals/alignment/alignment.json",
    "action/sandbox/sandbox.json",
    "goals/escalation/escalation.json",
]
MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
OCI_IMAGE_BY_DIGEST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}$")


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
    """A validated .agi package with path-confined access to its components."""

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        temporary_directory: tempfile.TemporaryDirectory[str] | None = None,
    ):
        self.root = root.resolve()
        self.manifest = manifest
        self._module_cache: dict[str, dict[str, Any]] = {}
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
        relative = _safe_relative_path(relative_path, "package reference")
        candidate = self.root / Path(*relative.parts)
        current = candidate
        while current != self.root:
            if current.is_symlink():
                raise AGILoadError(f"Package reference may not use symlinks: {relative_path}")
            current = current.parent
        full_path = candidate.resolve()
        if not _is_within(full_path, self.root):
            raise AGILoadError(f"Package reference escapes root: {relative_path}")
        return full_path

    def load_json(self, relative_path: str) -> dict[str, Any]:
        """Load an object-valued JSON file confined to the package root."""
        full_path = self.resolve_path(relative_path)
        if not full_path.is_file():
            raise AGILoadError(f"Missing file: {relative_path}")
        try:
            value = json.loads(full_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AGILoadError(f"Invalid JSON in {relative_path}: {exc}") from exc
        if not isinstance(value, dict):
            raise AGILoadError(f"Expected a JSON object in {relative_path}")
        return value

    def load_module(self, module_name: str) -> dict[str, Any]:
        if module_name in self._module_cache:
            return self._module_cache[module_name]
        capability = self.manifest["capabilities"].get(module_name)
        if not capability:
            raise AGILoadError(f"Unknown module: {module_name}")
        module_data = self.load_json(capability["module"])
        self._module_cache[module_name] = module_data
        return module_data

    def load_component(self, module_name: str, component_name: str) -> dict[str, Any]:
        module = self.load_module(module_name)
        for component in module.get("components", []):
            if component["name"] == component_name:
                return self.load_json(f"{module_name}/{component['path']}")
        raise AGILoadError(f"Component {component_name} not found in module {module_name}")

    def load_goal_loop(self) -> dict[str, Any]:
        return self.load_json(self.manifest["runtime"]["entrypoint"])

    def close(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> AGIPackage:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _validate_safety(package: AGIPackage) -> None:
    alignment = package.load_json(REQUIRED_SAFETY_FILES[0])
    constraints = alignment.get("config", {}).get("constraints")
    if not isinstance(constraints, list) or not constraints:
        raise AGILoadError("alignment.json must declare at least one constraint")
    constraint_ids = [item.get("id") for item in constraints if isinstance(item, dict)]
    if len(constraint_ids) != len(constraints) or len(set(constraint_ids)) != len(constraint_ids):
        raise AGILoadError("alignment.json constraint IDs must be present and unique")
    if not any(item.get("type") == "hard" for item in constraints):
        raise AGILoadError("alignment.json must contain at least one hard constraint")

    sandbox = package.load_json(REQUIRED_SAFETY_FILES[1])
    sandbox_config = sandbox.get("config", {})
    if sandbox_config.get("network", {}).get("policy") != "allowlist":
        raise AGILoadError("sandbox network policy must be allowlist")
    if not isinstance(sandbox_config.get("resource_limits"), dict):
        raise AGILoadError("sandbox.json must declare resource_limits")
    execution = sandbox_config.get("execution", {})
    isolation = sandbox_config.get("isolation", {})
    isolation_type = isolation.get("type")
    if isolation_type not in {"disabled", "local", "oci"}:
        raise AGILoadError("sandbox isolation type must be disabled, local, or oci")
    if isolation.get("fallback") != "deny":
        raise AGILoadError("sandbox isolation fallback must be deny")
    process_tools_enabled = bool(
        execution.get("shell_enabled", False) or execution.get("python_enabled", False)
    )
    if isolation_type == "disabled" and process_tools_enabled:
        raise AGILoadError("disabled process isolation cannot enable shell or Python")
    if isolation_type == "oci":
        if isolation.get("runtime") not in {"docker", "podman"}:
            raise AGILoadError("OCI isolation runtime must be docker or podman")
        if not OCI_IMAGE_BY_DIGEST.fullmatch(str(isolation.get("image", ""))):
            raise AGILoadError("OCI isolation image must be pinned by sha256 digest")

    escalation = package.load_json(REQUIRED_SAFETY_FILES[2])
    triggers = escalation.get("config", {}).get("escalation_triggers")
    if not isinstance(triggers, list) or not triggers:
        raise AGILoadError("escalation.json must declare escalation triggers")
    trigger_ids = [item.get("id") for item in triggers if isinstance(item, dict)]
    if len(trigger_ids) != len(triggers) or len(set(trigger_ids)) != len(trigger_ids):
        raise AGILoadError("escalation trigger IDs must be present and unique")


def _validate_package(package: AGIPackage) -> None:
    manifest = package.manifest
    _validate_schema(manifest, "manifest-v0.1.0.json", "manifest.json")
    version = manifest.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_FORMAT_VERSIONS))
        raise AGILoadError(f"Unsupported format_version {version!r}; supported: {supported}")

    registry = _load_schema("interfaces-v0.1.0.json")
    known_interfaces = set(registry["interfaces"])

    for module_name in REQUIRED_MODULES:
        capability = manifest["capabilities"][module_name]
        expected_path = f"{module_name}/module.json"
        if capability["module"] != expected_path:
            raise AGILoadError(
                f"Capability {module_name} must reference {expected_path}, not {capability['module']}"
            )
        module = package.load_json(expected_path)
        _validate_schema(module, "module-v0.1.0.json", expected_path)
        if module["name"] != module_name:
            raise AGILoadError(f"{expected_path}: name must be {module_name!r}")
        unknown_dependencies = set(module["dependencies"]) - set(REQUIRED_MODULES)
        if unknown_dependencies:
            raise AGILoadError(
                f"{expected_path}: unknown dependencies: {sorted(unknown_dependencies)}"
            )
        component_names: set[str] = set()
        for component in module["components"]:
            if component["name"] in component_names:
                raise AGILoadError(
                    f"{expected_path}: duplicate component {component['name']!r}"
                )
            component_names.add(component["name"])
            if component["interface"] not in known_interfaces:
                raise AGILoadError(
                    f"{expected_path}: unknown interface {component['interface']!r}"
                )
            component_path = f"{module_name}/{component['path']}"
            component_config = package.load_json(component_path)
            if component_path != manifest["runtime"]["entrypoint"]:
                _validate_schema(
                    component_config, "component-v0.1.0.json", component_path
                )
            if component_config["name"] != component["name"]:
                raise AGILoadError(
                    f"{component_path}: name does not match module component declaration"
                )

    for safety_file in REQUIRED_SAFETY_FILES:
        if not package.resolve_path(safety_file).is_file():
            raise AGILoadError(f"Missing required safety file: {safety_file}")
    _validate_safety(package)

    entrypoint = manifest["runtime"]["entrypoint"]
    goal_loop = package.load_json(entrypoint)
    _validate_schema(goal_loop, "goal-loop-v0.1.0.json", entrypoint)


def load_agi(path: str | Path) -> AGIPackage:
    """Load and fully validate a .agi directory or compressed archive."""
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

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        if temporary_directory:
            temporary_directory.cleanup()
        raise AGILoadError("Missing manifest.json in .agi package")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if temporary_directory:
            temporary_directory.cleanup()
        raise AGILoadError(f"Invalid manifest.json: {exc}") from exc
    if not isinstance(manifest, dict):
        if temporary_directory:
            temporary_directory.cleanup()
        raise AGILoadError("manifest.json must contain a JSON object")

    package = AGIPackage(root, manifest, temporary_directory)
    try:
        _validate_package(package)
    except Exception:
        package.close()
        raise

    log.info(
        "agi_loaded",
        name=package.name,
        version=package.version,
        format_version=package.format_version,
        modules=REQUIRED_MODULES,
    )
    return package
