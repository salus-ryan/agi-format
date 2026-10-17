# `.agi` agent blueprint format

[![CI](https://github.com/salus-ryan/agi-format/actions/workflows/ci.yml/badge.svg)](https://github.com/salus-ryan/agi-format/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`.agi` is a versioned, inspectable archive format for describing an AI agent's
capabilities, interfaces, safety policy, runtime requirements, and optional model
artifacts. This repository contains the **v0.1.0 format**, a conforming skeleton,
CLI tooling, and an **alpha reference runtime** backed by Qwen3-8B on Modal.

> This project is an experimental agent format and runtime. It is not evidence of
> artificial general intelligence, and the reference subprocess policy is not a
> substitute for a container or microVM security boundary.

**Deployment liveness:** <https://salus--agi-agent-agent.us-east.modal.run/health>

## Format overview

A `.agi` file is a deterministic `tar.gz` archive containing seven required
capability modules:

| Capability | Contract |
|---|---|
| Perception | Convert supported inputs into typed percepts |
| Reasoning | Inference, planning, world modeling, and verification |
| Memory | Episodic, semantic, working, and procedural stores |
| Language | Understanding, generation, formal languages, and user modeling |
| Action | Tool registry, execution policy, and feedback |
| Learning | Replay, curriculum, consolidation, and optional adaptation |
| Goals | Goal loop, decomposition, alignment, and escalation |

The format and runtime are deliberately separate. A package declares behavior and
constraints; a conforming host decides how interfaces are implemented. The Python
reference runtime currently consumes package policy for tools, sandbox limits,
planner bounds, replay capacity, alignment, and escalation. It does not provide a
general third-party plugin loader yet.

## Install

Python 3.12 or newer is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,api]'
```

Optional groups are `verification`, `perception`, and `learning`. GPU learning
dependencies are intentionally not part of the default installation.

## CLI

```bash
# Scaffold an independently authored, deny-by-default package
agi init my-agent --author "Example Organization"

# Full schema and interface-conformance validation
agi validate my-agent/

# Emit portable conformance evidence
agi validate my-agent/ --json --report conformance.json

# Inspect architecture and safety policy
agi inspect my-agent/

# Produce a deterministic archive
agi pack my-agent/ -o my-agent.agi

# Safely unpack it (links, devices, traversal, and archive bombs are rejected)
agi unpack my-agent.agi -o unpacked/
```

The canonical schemas and interface registry are in [`schemas/`](schemas/):

- `manifest-v0.1.0.json`
- `module-v0.1.0.json`
- `component-v0.1.0.json`
- `goal-loop-v0.1.0.json`
- `interfaces-v0.1.0.json`
- `conformance-report-v0.1.0.json`

The loader supports format version `0.1.0` exactly and fails closed on unknown
versions, interfaces, unsafe references, missing policies, or malformed modules.
Published schema identifiers are pinned to the immutable software release containing
them rather than the mutable `main` branch.

See [`docs/AUTHORING.md`](docs/AUTHORING.md) for the external-author workflow and
[`action.yml`](action.yml) for reusable GitHub Actions validation.

## Reference runtime

The reference pipeline is:

```text
perceive -> ground -> remember -> plan -> verify -> align -> act -> learn
```

It includes Qwen3-8B inference through vLLM, session-scoped memory, candidate plan
search, a causal/learned world model, a semantic knowledge graph, formal checks,
and prioritized replay.

Package configuration is used to bound planner candidates, replay capacity, enabled
tools, subprocess resources, filesystem access, network destinations, alignment,
and escalation. The default skeleton disables shell, Python, and outbound network
access. File operations are confined to each session workspace.

### Run the API locally

Start an OpenAI-compatible model server, then:

```bash
export AGI_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export AGI_PACKAGE_PATH="$PWD/skeleton"
export MEMORY_PATH="$PWD/.agent-memory"
export VLLM_BASE_URL="http://127.0.0.1:8000"
uvicorn runtime.api_server:app --host 127.0.0.1 --port 8080
```

All agent and state endpoints require a bearer token. `/health` and `/ready` are
public operational probes.

```bash
curl http://127.0.0.1:8080/chat \
  -H "Authorization: Bearer $AGI_API_KEY" \
  -H "X-Session-ID: demo" \
  -H "Content-Type: application/json" \
  -d '{"message":"What can you do?","max_tokens":256}'
```

Session IDs isolate conversation history, memories, goals, workspaces, and learning
state. Requests in one session are serialized. Deployments should use one container
per persistent store or replace the reference JSON stores with transactional storage.

### API endpoints

| Endpoint | Auth | Purpose |
|---|---:|---|
| `GET /health` | No | Process liveness |
| `GET /ready` | No | Package and model readiness |
| `POST /chat` | Yes | Run one agent step |
| `GET /status` | Yes | Session runtime status |
| `GET /memories` | Yes | Session episodic memories |
| `GET /action_log` | Yes | Session tool audit trail |
| `GET /world_model` | Yes | Session world-model summary |
| `GET /knowledge` | Yes | Session knowledge summary |
| `GET /dynamics` | Yes | Learned transition summary |
| `GET /learning` | Yes | Replay and adaptation status |
| `POST /train` | Yes | Trigger adaptation for the reserved `operator` session when explicitly enabled |

Request bodies and session IDs are bounded and validated. The API uses deny-by-default
CORS, bearer authentication, per-credential rate limiting, bounded session caches, and
redacted error responses.

## Action security

The reference tool runner provides defense in depth:

- no `shell=True`;
- command allowlists and shell-operator rejection;
- a minimal subprocess environment with no inherited credentials;
- filesystem confinement using resolved paths;
- CPU, memory, file, process, and descriptor limits on supported POSIX hosts;
- deny-by-default networking with domain allowlists and private-address blocking;
- no redirects, arbitrary ports, or POST requests by default;
- Python execution disabled by default.

For hostile workloads, place the entire runtime inside an additional container,
microVM, or equivalent boundary with an egress proxy. See [`SECURITY.md`](SECURITY.md).

## Learning safety

Retrieval learning and replay curation are active. Parametric LoRA updates are
**disabled by default** and require `AGI_ENABLE_ONLINE_TRAINING=true`.

When enabled, the trainer:

1. selects high-reward, deduplicated, secret-filtered examples;
2. creates a deterministic held-out split;
3. refuses to train without at least 6 GB of free GPU memory;
4. trains into a staging directory;
5. rejects validation-loss regressions;
6. versions the previous adapter;
7. promotes only after vLLM accepts the candidate; and
8. rolls back and writes an audit record on failure.

Operators should still review training data and evaluation evidence before enabling
weight updates in a sensitive deployment.

## Development

```bash
pytest -q
ruff check .
python -m build
python -m twine check dist/*
python scripts/smoke_deployment.py --url http://127.0.0.1:8080 --api-key "$AGI_API_KEY"
```

CI runs conformance tests, security regressions, API tests, a clean package build,
and installation from the built wheel. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Project structure

```text
schemas/             Published JSON Schemas, report schema, and interface registry
skeleton/            Conforming v0.1.0 starter blueprint bundled by `agi init`
docs/                 External authoring and conformance guidance
runtime/              Python loader, conformance reporter, and reference runtime
scripts/              Deployment smoke checks
tests/                Unit, integration, security, API, and conformance tests
action.yml            Reusable package-validation GitHub Action
agi_cli.py             init / validate / inspect / pack / unpack / run
modal_app.py          Authenticated Modal deployment
SPEC.md               Normative format specification
```

## License

MIT. See [`LICENSE`](LICENSE).
