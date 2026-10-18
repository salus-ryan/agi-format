# `.agi` agent boundary policy

[![CI](https://github.com/salus-ryan/agi-format/actions/workflows/ci.yml/badge.svg)](https://github.com/salus-ryan/agi-format/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`.agi` is a small, signed, portable file that says **what an agent is allowed to touch**:
which tools it may call, which need a human, which network destinations and processes are
permitted, and how many actions it may take. A host enforces that policy *before* every
effect, using deterministic checks only: no LLM judges and no keyword guesses.

The format deliberately says nothing about *how* an agent thinks. Models, prompts, memory,
planning, and learning are up to the agent. The policy defines the boundary it runs inside.

> Format 0.2.0 replaces the 0.1.0 "seven capability modules" blueprint and its reference
> agent runtime. That code is preserved at the [`v0.4.0`](https://github.com/salus-ryan/agi-format/tree/v0.4.0) tag.

## A package

```text
my-agent/
├── manifest.json     # identity and format version
├── policy.json       # the boundary (tools, budget, sandbox, escalation)
├── signature.json    # optional Ed25519 signature over every other file
└── ...               # any other files (prompts, weights refs); opaque, but covered by the digest
```

```json
{
  "tools": [
    {"name": "file_read", "scope": "fs:read"},
    {"name": "file_write", "scope": "fs:write", "requires_approval": true}
  ],
  "budget": {"max_actions_per_session": 100},
  "sandbox": {
    "network": {"allowed_domains": [], "allow_http": false, "allow_post": false},
    "execution": {"shell_enabled": false, "python_enabled": false, "allowed_commands": []},
    "isolation": {"type": "disabled", "fallback": "deny"},
    "resource_limits": {"max_cpu_seconds_per_action": 30, "max_memory_mb": 512}
  },
  "escalation": {
    "triggers": [{
      "id": "credential_paths",
      "description": "Reading or writing files that look like credentials needs a human.",
      "severity": "high",
      "tools": ["file_read", "file_write"],
      "params": {"path": "(?i)(\\.env|\\.pem|id_rsa|credentials|secret)"}
    }]
  }
}
```

Every call gets one verdict: **allow**, **deny** (the tool isn't declared, or the params are malformed), or
**escalate** (the tool requires approval, or a trigger's tool and parameter patterns match). If an
escalation has no approver, or the approver doesn't explicitly say yes, it is denied.

## Install

Python 3.12 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

## CLI

```bash
agi init my-agent --author "Example Organization"    # deny-by-default starter
agi validate my-agent/                                 # schema + policy semantics
agi check my-agent/ file_read --params '{"path": ".env"}'   # dry-run one decision
agi inspect my-agent/                                  # human-readable boundary

agi keygen --private-key signing.pem --public-key signing.pub
agi sign my-agent/ --key signing.pem
agi pack my-agent/ -o my-agent.agi                     # deterministic archive
agi validate my-agent.agi --trusted-key signing.pub --json --report conformance.json
agi unpack my-agent.agi -o unpacked/
```

`agi check` exits `0` for allow and `2` for deny or escalate.

## Enforcing a policy in a host

```python
from runtime.loader import load_agi
from runtime.policy import PolicyEnforcer
from runtime.signing import load_public_key

package = load_agi("my-agent.agi", trusted_keys=[load_public_key("signing.pub")])
executor = package.policy.build_executor("/srv/agent/workspace")   # reference built-in tools

def approve(decision, params) -> bool:
    return ask_a_human(decision.to_dict(), params)   # must return exactly True to allow

enforcer = PolicyEnforcer(package.policy, executor.execute, approve)
result = enforcer.execute("file_write", {"path": "notes.txt", "content": "hi"})
```

`PolicyEnforcer` works with any tool callable, so hosts can put their own tools (MCP servers,
APIs) behind the same policy. Each tool's `scope` is meant to map onto the host's authority
model, for example a capability grant.

## Reference tool sandbox

The bundled executor implements `file_read`, `file_write`, `shell`, `python_eval`, and `web_request`:

- files are confined to the session workspace through resolved paths;
- commands run as argv only (never `shell=True`), against an allowlist that rejects shell
  operators, file-list/output options, and option values that point outside the workspace;
- `oci` isolation runs processes in Docker/Podman with a digest-pinned image, no network, a
  read-only root, dropped capabilities, and resource limits (see [`docs/OCI_ISOLATION.md`](docs/OCI_ISOLATION.md));
- HTTP is deny-by-default with domain allowlists, private-address blocking, no redirects, and
  connections pinned to the validated address to prevent DNS rebinding.

The `local` subprocess profile is for trusted development only. See [`SECURITY.md`](SECURITY.md).

## Specification

[`SPEC.md`](SPEC.md) is normative. Schemas live in [`schemas/`](schemas/), and authoring and CI
usage are in [`docs/AUTHORING.md`](docs/AUTHORING.md) and [`action.yml`](action.yml).

## License

MIT
