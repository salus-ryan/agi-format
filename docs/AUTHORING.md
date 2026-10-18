# Authoring and conformance

This guide is for package authors who do not depend on this source tree.
The format version and the Python implementation version are separate: `agi-format` 0.5.0
validates `.agi` format 0.2.0.

## Create a package

Install the released wheel and scaffold into a new directory:

```bash
python -m venv .venv
. .venv/bin/activate
pip install https://github.com/salus-ryan/agi-format/releases/download/v0.5.0/agi_format-0.5.0-py3-none-any.whl
agi init my-agent \
  --name my-agent \
  --author "Example Organization" \
  --description "A narrowly scoped example agent" \
  --license MIT
agi validate my-agent
```

`agi init` fails rather than overwriting an existing path. The generated policy allows only
workspace file reads, requires approval for writes, and disables processes and network access.
Relaxing those controls requires a host security review; a policy file is not an
operating-system sandbox on its own.

Edit `manifest.json` and `policy.json`, then dry-run decisions:

```bash
agi check my-agent file_read --params '{"path": "notes.txt"}'
agi inspect my-agent
```

The normative requirements are in [`../SPEC.md`](../SPEC.md). The format 0.2.0 schemas are
published under the `v0.5.0` release tag.

## Sign it

```bash
agi keygen --private-key signing.pem --public-key signing.pub
agi sign my-agent --key signing.pem
```

Sign after every edit; any change to a file other than `signature.json` invalidates the signature.
Distribute `signing.pub` to hosts out of band and keep `signing.pem` out of agent environments.

## Produce conformance evidence

```bash
agi validate my-agent --json --report conformance.json
agi pack my-agent -o my-agent.agi
agi validate my-agent.agi --trusted-key signing.pub --json --report archive-conformance.json
```

A successful report includes:

- validator implementation and version;
- declared format version;
- package name and version;
- a SHA-256 digest over sorted relative paths and exact file contents (excluding `signature.json`);
- file count and uncompressed byte count;
- the signing key fingerprint and whether it was in the trusted set;
- explicit structure, schema, policy-semantics, and signature checks.

The digest is the same for a directory and its deterministic `.agi` archive. A report establishes
that the boundary is well-formed and attributable; it is not an endorsement of the agent's behavior.

## Add validation to another repository

The repository provides a composite GitHub Action:

```yaml
name: .agi conformance
on: [push, pull_request]
permissions:
  contents: read
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: salus-ryan/agi-format@v0.5.0
        with:
          package: path/to/my-agent
          report: agi-conformance-report.json
      - uses: actions/upload-artifact@v6
        with:
          name: agi-conformance-report
          path: agi-conformance-report.json
```

Pin a release tag or full commit, not `main`.

## Submit an interoperability result

Open the repository's **Conformance pilot** issue form. Include the package repository and commit,
the report, and the independent runtime or tool being tested. Never include credentials, private
weights, customer data, or proprietary prompts.
