# Changelog

This project follows [Semantic Versioning](https://semver.org/) for the Python runtime.
The `.agi` format has an independent version declared in each package manifest.

## [Unreleased]

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
