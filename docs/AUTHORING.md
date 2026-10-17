# Authoring and conformance

This guide is for package authors who do not depend on the reference runtime's source tree.
The format version and the Python implementation version are separate: `agi-format` 0.3.0
validates `.agi` format 0.1.0.

## Create a package

Install the released wheel and scaffold into a new directory:

```bash
python -m venv .venv
. .venv/bin/activate
pip install https://github.com/salus-ryan/agi-format/releases/download/v0.3.0/agi_format-0.3.0-py3-none-any.whl
agi init my-agent \
  --name my-agent \
  --author "Example Organization" \
  --description "A narrowly scoped example agent" \
  --license MIT
agi validate my-agent
```

`agi init` fails rather than overwriting an existing path. The generated policy disables shell,
caller-supplied Python, unrestricted network access, dynamic tool discovery, and parametric online
learning. Relaxing those controls requires a host security review; package declarations are not an
operating-system sandbox.

Edit `manifest.json`, each capability's `module.json`, and component configuration files. The
normative requirements are in [`../SPEC.md`](../SPEC.md). The immutable format 0.1.0 schemas are
published under the [`v0.3.0`](https://github.com/salus-ryan/agi-format/tree/v0.3.0/schemas) release
tag.

## Produce conformance evidence

```bash
agi validate my-agent --json --report conformance.json
agi pack my-agent -o my-agent.agi
agi validate my-agent.agi --json --report archive-conformance.json
```

A successful report includes:

- validator implementation and version;
- declared format version;
- package name and version;
- a SHA-256 digest over sorted relative paths and exact file contents;
- file count and uncompressed byte count;
- explicit structure, schema, interface-registry, and safety-profile checks.

The digest is the same for a directory and its deterministic `.agi` archive. A report establishes
syntactic and policy-profile conformance only; it is not an endorsement of the package's behavior,
security, or model quality.

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
      - uses: salus-ryan/agi-format@v0.3.0
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
