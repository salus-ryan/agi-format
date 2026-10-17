# Security policy

## Supported versions

| Version | Security fixes |
|---|---|
| `main` / latest `0.x` runtime | Yes |
| Earlier runtime revisions | Critical fixes only |
| `.agi` format `0.1.0` | Current |

## Reporting a vulnerability

Please do **not** open a public issue for a suspected vulnerability. Use GitHub's
**Report a vulnerability** flow under the repository Security tab. Include:

- affected commit or deployment;
- minimal reproduction;
- expected impact;
- whether credentials, private data, or external actions are involved; and
- any suggested mitigation.

Do not access data that is not yours, persist in deployed systems, or perform
destructive testing. We will acknowledge a complete report as soon as practical,
coordinate remediation, and credit reporters who want attribution.

## Threat model

The runtime treats all user input, model output, package content, memories, and web
content as untrusted. Important boundaries are:

1. archive/package parsing;
2. API authentication and session selection;
3. model-generated tool calls;
4. filesystem and network access;
5. persisted memory and training data; and
6. adapter promotion.

The reference implementation rejects archive traversal and links, validates package
schemas and interfaces, requires bearer authentication for stateful endpoints,
confines sessions, strips process environments, bounds output while draining child pipes,
uses tool and network allowlists, and disables shell, Python, networking, and parametric
training by default. Production mode refuses process tools without a configured OCI boundary.

## Deployment requirements

- Generate `AGI_API_KEY` with at least 256 bits of entropy and store it in a secret manager.
- Do not expose the internal vLLM port.
- Keep `AGI_REQUIRE_AUTH=true` outside local development.
- Set explicit CORS origins or leave CORS disabled.
- Keep one process/container per JSON-backed persistent store.
- Keep `AGI_REQUIRE_STRONG_TOOL_ISOLATION=true` in production.
- Leave process isolation `disabled`, or configure Docker/Podman with an image pinned by digest.
- Prefer a dedicated worker container or microVM over access to a shared container daemon for hostile workloads.
- Use an egress proxy and DNS controls for production network tools.
- Never mount cloud credentials, SSH keys, or unrelated secrets into a tool-enabled runtime.
- Leave `AGI_ENABLE_ONLINE_TRAINING=false` unless training data and evaluation are reviewed.
- Monitor action logs, authentication failures, rate limits, and adapter audit records.

## Known limitations

The local subprocess policy is for trusted development, not a hardened kernel boundary.
The OCI backend depends on the host runtime, kernel, image, and daemon configuration; it is
not equivalent to a microVM and deliberately provides no process network access. HTTP domain
validation alone cannot eliminate every rebinding or proxy ambiguity. The in-process rate
limiter does not coordinate across replicas. The bundled JSON persistence is intended for a
single container and should be replaced with transactional storage before horizontal scaling.
