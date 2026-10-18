# OCI process-tool isolation

The reference executor can run allowlisted command and Python tools inside Docker or Podman.
This backend is opt-in. The starter package declares process isolation `disabled`, so it runs
neither shell commands nor caller-supplied Python.

OCI isolation is a stronger boundary than a same-process subprocess profile, but it is not a
microVM. The host kernel, container runtime, daemon, pinned image, and workspace contents remain
trusted computing-base components.

## Prepare an image

Build, scan, and sign a minimal image outside the agent runtime. Resolve it to an immutable local
repository digest:

```bash
docker pull example.invalid/agi-tools:1.0
docker image inspect --format '{{index .RepoDigests 0}}' example.invalid/agi-tools:1.0
# example.invalid/agi-tools@sha256:<64 lowercase hex characters>
```

The runtime uses `--pull=never`; the exact digest must already exist locally. A tag such as
`:latest` is rejected. Never mount the Docker/Podman socket inside the tool container.

## Package policy

Configure the `sandbox` section of `policy.json` and declare the `shell` tool:

```json
{
  "tools": [{"name": "shell", "scope": "process:exec"}],
  "sandbox": {
    "execution": {
      "shell_enabled": true,
      "python_enabled": false,
      "allowed_commands": ["pwd", "ls", "head", "wc"]
    },
    "isolation": {
      "type": "oci",
      "runtime": "docker",
      "image": "example.invalid/agi-tools@sha256:<digest>",
      "fallback": "deny"
    }
  }
}
```

Sandbox flags cannot add an undeclared tool. Keep `require_strong_isolation=True` when calling
`Policy.build_executor`; if the OCI setup is unavailable or invalid, executor construction fails
instead of falling back to local execution.

## Enforced invocation

The host owns all container-runtime arguments. A model-generated command cannot alter them. Each
execution receives:

- `--network=none`;
- `--read-only` root filesystem;
- `--cap-drop=ALL` and `no-new-privileges`;
- one CPU plus package-bounded memory and 16-process ceilings;
- a small `noexec,nosuid` temporary filesystem;
- the numeric host user and group, not container root;
- only the session workspace mounted at `/workspace`;
- bounded stdout and stderr capture; and
- a generated container name used for forced cleanup after timeout.

Command paths outside the workspace are rejected before invocation. Process environment variables
are reconstructed from a minimal allowlist and do not inherit application credentials.

## Remaining boundaries

- HTTP tools execute in the host runtime under a separate deny-by-default domain policy; they are
  not enabled inside process containers.
- A compromised container runtime or host kernel defeats this boundary.
- Workspace files are intentionally writable and must be treated as untrusted after execution.
- Multi-tenant hostile workloads should use a dedicated worker container or microVM rather than a
  shared daemon.
