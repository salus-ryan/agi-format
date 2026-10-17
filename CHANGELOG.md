# Changelog

This project follows [Semantic Versioning](https://semver.org/) for the Python runtime.
The `.agi` format has an independent version declared in each package manifest.

## [Unreleased]

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
