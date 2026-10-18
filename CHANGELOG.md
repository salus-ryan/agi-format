# Changelog

This project follows [Semantic Versioning](https://semver.org/) for the Python runtime.
The `.agi` format has an independent version declared in each package manifest.

## [Unreleased]

## [0.5.0]

`.agi` format 0.2.0 is a breaking redesign. The package is now a boundary policy, not a
cognitive-architecture blueprint. The 0.1.0 format and the agent runtime remain available at `v0.4.0`.

### Added

- `policy.json`: declared tools with scopes and approval flags, a per-session action budget,
  sandbox limits, and deterministic escalation triggers with parameter patterns.
- `runtime.policy`: `Policy.evaluate` (allow/deny/escalate) and `PolicyEnforcer`, which applies
  the policy before every effect and fails closed without an explicit approval.
- Ed25519 package signing (`agi keygen`, `agi sign`, `--trusted-key`) over a content digest
  that excludes `signature.json`.
- `agi check` to dry-run one decision against a package.

### Removed

- The seven required capability modules, the interface registry, `goal_loop.json`, and the
  alignment constraints judged by an LLM.
- The reference agent runtime (LLM kernel, planner, world model, memory, learning, LoRA,
  perception), the FastAPI server, the Modal deployment, and `agi run`.

### Security

- Shell arguments may no longer reach host files through `--files0-from`, `--output`, or
  option values (`--opt=/path`, `-o/path`, `~`) outside the workspace.
- `web_request` connects to the address it validated (with the original Host header and
  SNI), so a second DNS answer can't redirect it to a private address.
- Any resolution that includes a non-global address is rejected.

## [0.4.0] - 2026-10-17

### Added

- Opt-in Docker/Podman process isolation with digest-pinned images, no network,
  read-only roots, dropped capabilities, no-new-privileges, and resource limits.
- Bounded streaming capture for child stdout and stderr, including explicit
  truncation evidence and process-group termination on timeout.
- Forced OCI container cleanup after an attached client timeout.

### Changed

- The starter package now truthfully declares process execution as disabled.
- Production mode fails closed if a package enables shell or Python without OCI isolation.
- Empty tool and command allowlists now remain empty instead of selecting defaults.
- Sandbox toggles can no longer enable tools absent from the package tool registry.

## [0.3.0] - 2026-10-17

### Added

- `agi init` for generating independently authored, deny-by-default starter packages.
- Machine-readable, schema-validated conformance reports with stable content digests.
- A reusable composite GitHub Action and conformance-pilot issue form.
- External package-authoring and interoperability guidance.

### Changed

- Canonical schema identifiers now target the immutable `v0.3.0` release tag.
- The skeleton is bundled in wheels so scaffolding works outside a source checkout.
- The API reports its version from the installed runtime package.

## [0.2.0] - 2026-10-17

### Added

- Published JSON Schemas and an interface registry for format v0.1.0.
- Deep conformance checks, secure archive extraction, and deterministic packing.
- Bearer authentication, session isolation, rate limiting, bounded requests, and readiness checks.
- Deny-by-default tool execution and outbound-network policy.
- Evaluation-gated, versioned, auditable LoRA promotion and rollback.
- CI, deployment smoke checks, security policy, contribution guide, and MIT license.

### Changed

- Reference runtime configuration now consumes package tool, sandbox, planner, replay,
  alignment, and escalation settings.
- Parametric online training is disabled by default.
- Public health responses no longer expose session state.

### Fixed

- Compute reward before passing it to goal tracking, removing the live `/chat`
  `UnboundLocalError`.
- Declare complete core, API, development, verification, perception, and learning dependencies.
- Make editable and wheel installation work with explicit package discovery.

### Security

- Removed `shell=True`, inherited subprocess secrets, unrestricted outbound requests,
  wildcard CORS, unauthenticated state endpoints, and shared cross-session kernels.

## [0.1.0] - 2026-10-16

- Initial format specification, skeleton package, Python runtime, and Modal demo.
