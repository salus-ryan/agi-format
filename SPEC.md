# `.agi` Format Specification v0.2.0

## Purpose

An `.agi` package is a portable, optionally signed declaration of the boundary an AI agent must
operate within. It specifies which effects the agent may cause, which require human approval,
and the sandbox limits for each effect. It does not describe the agent's cognition.

The key words MUST, MUST NOT, SHOULD, and MAY are to be interpreted as described in RFC 2119.

## Design principles

1. **Enforceable only.** Every field changes a deterministic allow/deny/escalate decision or a
   sandbox limit. A field that a host cannot check mechanically does not belong in the format.
2. **Fail closed.** Unknown versions, fields, tools, malformed parameters, missing approvers,
   and invalid signatures all result in refusal.
3. **Enforced before the effect.** A host MUST evaluate the policy before it executes a call,
   not afterwards on the model's output.
4. **Cognition-agnostic.** Models, prompts, memory, and learning are out of scope. Packages MAY
   carry such files, but conforming hosts treat them as opaque content covered by the digest.

## Package layout

A package is a directory, or a deterministic `tar.gz` archive of one, with the extension `.agi`.

| Path | Required | Schema |
|---|---|---|
| `manifest.json` | Yes | `schemas/manifest-v0.2.0.json` |
| `policy.json` | Yes | `schemas/policy-v0.2.0.json` |
| `signature.json` | No | `schemas/signature-v0.2.0.json` |
| any other file | No | opaque |

Packages MUST NOT contain symbolic links, hard links, devices, FIFOs, absolute paths, or `..`
segments. Archives MUST also be rejected if they contain duplicate paths, more than 10,000 members,
or more than 2 GiB of expanded content.

## manifest.json

```json
{
  "$schema": "https://raw.githubusercontent.com/salus-ryan/agi-format/v0.5.0/schemas/manifest-v0.2.0.json",
  "format_version": "0.2.0",
  "agent": {"name": "my-agent", "version": "1.0.0", "author": "...", "license": "MIT"}
}
```

A host MUST reject any `format_version` it does not explicitly support. Version ranges are not
supported.

## policy.json

### tools

Each entry declares a tool the agent may request. Any tool not listed MUST be denied.

| Field | Meaning |
|---|---|
| `name` | Tool identifier, unique within the policy. |
| `scope` | Authority scope `namespace:action`, defaulting to `tool:<name>`. Hosts SHOULD map it onto their authorization model. |
| `requires_approval` | If true, every call escalates. |

`file_read`, `file_write`, `shell`, `python_eval`, and `web_request` are the reference built-in
tools and carry the sandbox semantics below. Any other name is a host-defined tool that is gated
by the same verdicts.

### budget

`max_actions_per_session` caps the number of executed (not refused) calls per enforcer
instance. If absent, there is no cap.

### sandbox

| Field | Meaning |
|---|---|
| `network.allowed_domains` | Hostnames (and their subdomains) `web_request` may reach. Empty means no network. |
| `network.allow_http` | Permit plaintext HTTP. Default false. |
| `network.allow_post` | Permit POST. Default false (GET/HEAD only). |
| `execution.shell_enabled` / `python_enabled` | Permit process tools. |
| `execution.allowed_commands` | argv[0] allowlist for `shell`. |
| `isolation.type` | `disabled`, `local`, or `oci`. |
| `isolation.runtime` / `image` | For `oci`: `docker` or `podman`, and an image pinned by `@sha256:`. |
| `isolation.fallback` | MUST be `deny`. A missing runtime or image is an error, never permission to run locally. |
| `resource_limits` | Per-action CPU seconds (1–60) and memory (64–2048 MB). |

The following cross-field rules are conformance errors:

- `disabled` isolation together with shell or Python enabled;
- `oci` isolation without a runtime or a digest-pinned image;
- `shell` declared without `shell_enabled` or with no allowed commands;
- `python_eval` declared without `python_enabled`;
- `web_request` declared with no allowed domains.

Reference hosts MUST connect to the address they validated rather than resolving the hostname a
second time, and MUST reject redirects and any resolution that includes a non-global address.

### escalation

Each trigger has an `id`, `description`, `severity` (`low`, `medium`, `high`, or `critical`),
`tools` (declared tool names or `"*"`), and optional `params`. `params` maps parameter names to
regular expressions. A trigger matches when the tool matches and **every** listed parameter is
present and its value (or its JSON encoding, if not a string, truncated to 4096 characters)
contains a match for its pattern. Triggers MUST reference only declared tools, and patterns MUST
compile.

## Decision procedure

For a proposed call `(tool, params)`:

1. If `tool` is not declared: **deny**.
2. If `params` is not an object: **deny**.
3. If the tool has `requires_approval`: **escalate**.
4. If any trigger matches, the first one in declaration order: **escalate** with its id and severity.
5. Otherwise: **allow**.

On **escalate**, a host MUST obtain an explicit affirmative approval before executing. A missing
approver, an error, a timeout, or any answer other than an explicit yes MUST be treated as a
denial. After this, if the action budget is exhausted the call is denied; otherwise it is
executed and counted.

## Content digest and signature

The content digest is a SHA-256 over every regular file except a root `signature.json`, in
ascending order of package-relative POSIX path. For each file the hash receives: the path's
UTF-8 byte length as a 4-byte big-endian integer, the UTF-8 path, the file size as an 8-byte
big-endian integer, and the file's bytes. A directory and its archive produce the same digest.

`signature.json` contains an Ed25519 public key (base64, 32 bytes), the digest, and a signature
over `"agi-format package signature v0.2.0\n"` followed by the lowercase hex digest. A host MUST
reject a package whose signature document is present but malformed, does not match the
recomputed digest, or does not verify. A host that was given a set of trusted keys MUST also
reject unsigned packages and packages signed by any other key.

## Conformance

A package conforms when it passes the schemas, the cross-field rules, path and archive safety,
and, if present, signature verification. `agi validate --json` emits the report defined by
`schemas/conformance-report-v0.2.0.json`. Conformance does not certify an agent's behavior; it
certifies only that the boundary is well-formed and attributable.

Schema identifiers MUST refer to immutable release tags, never to `main`.

## MIME type and extension

`application/x-agi`, `.agi`.
