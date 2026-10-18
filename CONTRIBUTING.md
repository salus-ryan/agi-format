# Contributing

## Setup

```bash
git clone https://github.com/salus-ryan/agi-format.git
cd agi-format
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

## Required checks

```bash
pytest -q
ruff check .
agi validate skeleton/
python -m build
python -m twine check dist/*
```

Changes to schemas, policy evaluation, signing, the loader, or tool execution must
include positive and negative tests.

## Format changes

1. Keep `SPEC.md`, schemas, skeleton files, and loader behavior synchronized.
2. Add malformed-package tests for every new invariant.
3. Preserve deterministic packing.
4. Do not silently accept unknown format versions or fields.
5. Add a field only if the reference enforcer checks it mechanically.
6. Use a new format version for incompatible changes.

## Security changes

Follow `SECURITY.md`. Do not weaken deny-by-default behavior to make a demo pass.
High-risk capabilities require an explicit package declaration, host permission, and
an isolation boundary. Never commit signing keys, API keys, or model tokens.

## Pull requests

Keep changes focused, explain the user-visible behavior, list verification commands,
and call out deployment or migration requirements. Update `CHANGELOG.md` under
`Unreleased`.
