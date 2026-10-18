# Security policy

## Supported versions

| Version | Security fixes |
|---|---|
| `main` / latest `0.x` release | Yes |
| Earlier releases | Critical fixes only |
| `.agi` format `0.2.0` | Current |
| `.agi` format `0.1.0` (`v0.4.0` and earlier) | Unsupported |

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

Package content, model output, tool parameters, and web content are all untrusted. The
important boundaries are:

1. archive and package parsing;
2. signature verification and key trust;
3. policy evaluation before every effect, including approval handling; and
4. the reference tool sandbox (filesystem, processes, network).

The reference implementation rejects archive traversal and links, validates schemas and
cross-field policy rules, verifies Ed25519 signatures over a content digest, makes every
decision deterministically from the policy, denies escalations without an explicit approval,
strips process environments, bounds output while draining child pipes, and pins outbound
connections to validated public addresses.

## Host requirements

- Pass `trusted_keys` (or `--trusted-key`) when loading packages from anyone you do not control.
- Keep signing keys out of agent environments. A key that can be read by the agent can widen its own boundary.
- Route every tool, including host-defined ones, through `PolicyEnforcer` or an equivalent check.
- Make approvers return `True` only after an explicit human decision. Treat timeouts as denials.
- Keep `require_strong_isolation=True`; leave isolation `disabled` or use OCI with a digest-pinned image.
- Prefer a dedicated worker container or microVM over a shared container daemon for hostile workloads.
- Never mount cloud credentials, SSH keys, or unrelated secrets into a tool-enabled host.

## Known limitations

The `local` subprocess profile is for trusted development, not a hardened kernel boundary.
The OCI backend depends on the host runtime, kernel, image, and daemon configuration, and is not
equivalent to a microVM. Trigger patterns are Python regular expressions applied to at most 4096
characters per parameter; authors should avoid catastrophic-backtracking patterns. The action
budget is per enforcer instance and does not coordinate across processes.
