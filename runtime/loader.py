"""Loader for .agi files -- unpacks and validates the archive."""

import json
import tarfile
import tempfile
from pathlib import Path

import structlog

log = structlog.get_logger()

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


class AGILoadError(Exception):
    """Raised when a .agi file is malformed or invalid."""


class AGIPackage:
    """Represents a loaded .agi file with access to all its components."""

    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self.manifest = manifest
        self._module_cache: dict[str, dict] = {}

    @property
    def name(self) -> str:
        return self.manifest["agent"]["name"]

    @property
    def version(self) -> str:
        return self.manifest["agent"]["version"]

    @property
    def format_version(self) -> str:
        return self.manifest["format_version"]

    def load_json(self, relative_path: str) -> dict:
        """Load a JSON file from the package."""
        full_path = self.root / relative_path
        if not full_path.exists():
            raise AGILoadError(f"Missing file: {relative_path}")
        with open(full_path) as f:
            return json.load(f)

    def load_module(self, module_name: str) -> dict:
        """Load a module manifest by name."""
        if module_name in self._module_cache:
            return self._module_cache[module_name]

        cap = self.manifest["capabilities"].get(module_name)
        if not cap:
            raise AGILoadError(f"Unknown module: {module_name}")

        module_data = self.load_json(cap["module"])
        self._module_cache[module_name] = module_data
        return module_data

    def load_component(self, module_name: str, component_name: str) -> dict:
        """Load a specific component config from a module."""
        module = self.load_module(module_name)
        for comp in module.get("components", []):
            if comp["name"] == component_name:
                return self.load_json(f"{module_name}/{comp['path']}")
        raise AGILoadError(
            f"Component {component_name} not found in module {module_name}"
        )

    def load_goal_loop(self) -> dict:
        """Load the main agent loop definition."""
        entrypoint = self.manifest["runtime"]["entrypoint"]
        return self.load_json(entrypoint)


def load_agi(path: str | Path) -> AGIPackage:
    """Load a .agi file (tar.gz archive) or an unpacked directory.

    Returns an AGIPackage with validated contents.
    """
    path = Path(path)

    if path.is_dir():
        root = path
        log.info("loading_agi_directory", path=str(path))
    elif path.suffix == ".agi" or path.suffixes[-2:] == [".tar", ".gz"]:
        root = Path(tempfile.mkdtemp(prefix="agi_"))
        log.info("extracting_agi_archive", path=str(path), to=str(root))
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(root, filter="data")
        # Handle case where archive has a single top-level directory
        children = list(root.iterdir())
        if len(children) == 1 and children[0].is_dir():
            root = children[0]
    else:
        raise AGILoadError(f"Unsupported file type: {path.suffix}")

    # Validate manifest
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise AGILoadError("Missing manifest.json in .agi package")

    with open(manifest_path) as f:
        manifest = json.load(f)

    _validate_manifest(manifest)

    package = AGIPackage(root, manifest)

    # Validate all required modules exist
    for module_name in REQUIRED_MODULES:
        module_dir = root / module_name
        if not module_dir.is_dir():
            raise AGILoadError(f"Missing required module directory: {module_name}/")
        module_json = module_dir / "module.json"
        if not module_json.exists():
            raise AGILoadError(f"Missing module manifest: {module_name}/module.json")

    # Validate required safety files
    for safety_file in REQUIRED_SAFETY_FILES:
        if not (root / safety_file).exists():
            raise AGILoadError(
                f"Missing required safety file: {safety_file}. "
                "A .agi file without safety constraints is malformed."
            )

    # Validate entrypoint exists
    entrypoint = manifest["runtime"]["entrypoint"]
    if not (root / entrypoint).exists():
        raise AGILoadError(f"Missing entrypoint: {entrypoint}")

    log.info(
        "agi_loaded",
        name=package.name,
        version=package.version,
        format_version=package.format_version,
        modules=REQUIRED_MODULES,
    )

    return package


def _validate_manifest(manifest: dict) -> None:
    """Validate the root manifest structure."""
    required_keys = ["format_version", "agent", "capabilities", "runtime"]
    for key in required_keys:
        if key not in manifest:
            raise AGILoadError(f"Manifest missing required key: {key}")

    agent = manifest["agent"]
    for key in ["name", "version"]:
        if key not in agent:
            raise AGILoadError(f"Manifest agent section missing: {key}")

    capabilities = manifest["capabilities"]
    for module_name in REQUIRED_MODULES:
        if module_name not in capabilities:
            raise AGILoadError(f"Manifest missing required capability: {module_name}")

    runtime = manifest["runtime"]
    if "entrypoint" not in runtime:
        raise AGILoadError("Manifest runtime section missing: entrypoint")
    if not runtime.get("sandbox_required", False):
        raise AGILoadError("sandbox_required must be true in manifest")
