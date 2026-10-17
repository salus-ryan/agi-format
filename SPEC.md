# .agi File Format Specification v0.1.0

**Status:** Alpha. This document is normative for format version `0.1.0`; the Python
loader and the JSON Schemas in `schemas/` are the conformance reference.

## Overview

A `.agi` file is a **tar.gz archive** containing a self-describing AGI agent blueprint.
When loaded by a conforming runtime, it instantiates an autonomous agent with seven
composable capabilities: perception, reasoning, memory, language, action, learning,
and goal management.

The format is declarative -- it describes *what* the agent is, not how to execute it.
A runtime interprets the spec and provides compute, I/O, and sandboxing.

## Design Principles

1. **Portability**: Archive is OS-agnostic. Runtime handles platform differences.
2. **Composability**: Each capability is a swappable module with a defined interface.
3. **Inspectability**: Everything is JSON/YAML + weights files. No opaque blobs without manifests.
4. **Incrementality**: You can ship a `.agi` with stub modules and progressively replace them.
5. **Safety by default**: Action sandboxing, value alignment constraints, and escalation rules are required fields, not optional.

## Archive Structure

```
agent.agi (tar.gz)
├── manifest.json                    # Root manifest (required)
├── perception/
│   ├── module.json                  # Perception module manifest
│   ├── adapters/
│   │   ├── text.json                # Text input adapter config
│   │   ├── vision.json              # Vision input adapter config
│   │   ├── audio.json               # Audio input adapter config
│   │   └── structured.json          # Structured data adapter config
│   └── grounding/
│       └── grounding.json           # Perceptual grounding config
├── reasoning/
│   ├── module.json                  # Reasoning module manifest
│   ├── inference/
│   │   └── engine.json              # Base inference engine config
│   ├── causal/
│   │   └── world_model.json         # Causal / world model config
│   ├── planning/
│   │   └── planner.json             # Hierarchical planner config
│   └── formal/
│       └── verifier.json            # Formal verification config
├── memory/
│   ├── module.json                  # Memory module manifest
│   ├── episodic/
│   │   └── episodic.json            # Episodic memory config
│   ├── semantic/
│   │   └── semantic.json            # Semantic knowledge config
│   ├── working/
│   │   └── working.json             # Working memory config
│   └── procedural/
│       └── procedural.json          # Procedural memory config
├── language/
│   ├── module.json                  # Language module manifest
│   ├── understanding/
│   │   └── nlu.json                 # NLU pipeline config
│   ├── generation/
│   │   └── nlg.json                 # NLG pipeline config
│   ├── formal_languages/
│   │   └── formal.json              # Code/math/logic config
│   └── theory_of_mind/
│       └── tom.json                 # Theory of mind config
├── action/
│   ├── module.json                  # Action module manifest
│   ├── tools/
│   │   └── tools.json               # Available tool definitions
│   ├── sandbox/
│   │   └── sandbox.json             # Execution sandbox config
│   └── feedback/
│       └── feedback.json            # Action-observation loop config
├── learning/
│   ├── module.json                  # Learning module manifest
│   ├── online/
│   │   └── online.json              # Online learning config
│   ├── replay/
│   │   └── replay.json              # Experience replay config
│   ├── curriculum/
│   │   └── curriculum.json          # Curriculum / self-play config
│   └── consolidation/
│       └── consolidation.json       # Anti-forgetting config
├── goals/
│   ├── module.json                  # Goals module manifest
│   ├── goal_loop.json               # Main agent loop definition
│   ├── alignment/
│   │   └── alignment.json           # Value alignment constraints
│   ├── decomposition/
│   │   └── decomposition.json       # Goal decomposition config
│   └── escalation/
│       └── escalation.json          # Human escalation rules
└── weights/                         # Optional: bundled model weights
    └── README.md                    # Weight manifest / download refs
```

## manifest.json (Root)

The root manifest is the entrypoint. It declares the agent's identity, version,
required capabilities, and runtime requirements.

```json
{
  "$schema": "https://raw.githubusercontent.com/salus-ryan/agi-format/v0.3.0/schemas/manifest-v0.1.0.json",
  "format_version": "0.1.0",
  "agent": {
    "name": "string (required)",
    "version": "string (semver, required)",
    "description": "string (optional)",
    "author": "string (optional)",
    "license": "string (optional)"
  },
  "capabilities": {
    "perception":  { "module": "perception/module.json",  "required": true },
    "reasoning":   { "module": "reasoning/module.json",   "required": true },
    "memory":      { "module": "memory/module.json",      "required": true },
    "language":    { "module": "language/module.json",     "required": true },
    "action":      { "module": "action/module.json",      "required": true },
    "learning":    { "module": "learning/module.json",     "required": true },
    "goals":       { "module": "goals/module.json",       "required": true }
  },
  "runtime": {
    "min_memory_gb": "number",
    "gpu_required": "boolean",
    "min_gpu_vram_gb": "number | null",
    "sandbox_required": true,
    "entrypoint": "goals/goal_loop.json"
  }
}
```

## Module Manifests (module.json)

Each capability directory has a `module.json` that declares its components,
dependencies, and interface contracts.

```json
{
  "name": "string",
  "version": "string (semver)",
  "description": "string",
  "components": [
    {
      "name": "string",
      "path": "string (relative to module dir)",
      "type": "adapter | engine | store | planner | tool | policy",
      "interface": "string (interface ID from interface registry)"
    }
  ],
  "dependencies": ["list of other module names this module requires"],
  "inputs": [
    { "name": "string", "type": "string (type ID)", "from": "module.component | environment" }
  ],
  "outputs": [
    { "name": "string", "type": "string (type ID)", "to": "module.component | environment" }
  ]
}
```

## Interfaces

Modules communicate through typed interfaces. Interface identifiers MUST be present
in `schemas/interfaces-v0.1.0.json`; unknown identifiers are a conformance error. The
core types are:

| Type ID               | Description                                    |
|-----------------------|------------------------------------------------|
| `Percept`             | Unified internal representation of a sensory input |
| `State`               | Current agent state (working memory contents)  |
| `Plan`                | Ordered list of steps toward a goal            |
| `Action`              | Executable action (tool call, API call, etc.)  |
| `Observation`         | Result of executing an action                  |
| `Goal`                | Objective with success criteria                |
| `Memory`              | Retrievable memory record                      |
| `Token`               | Language token (text, code, formal)            |
| `Embedding`           | Dense vector representation                    |
| `Reward`              | Scalar feedback signal                         |

## Conformance evidence

A package conforms when the reference validator accepts its manifest, all seven module
manifests, every declared component, the interface registry, the goal loop, and required
safety policy. `agi validate --json` emits the portable report defined by
`schemas/conformance-report-v0.1.0.json`.

The report's content digest processes files in ascending package-relative path order. For
each file, SHA-256 receives the path's four-byte unsigned big-endian UTF-8 byte length,
the UTF-8 path, the file's eight-byte unsigned big-endian size, and its exact bytes.
Directory and deterministic archive representations of the same package therefore produce
the same digest. Conformance does not certify model behavior, security, or runtime quality.

Schema identifiers MUST refer to immutable release tags. Package authors MUST NOT use the
mutable `main` branch as a normative schema identifier.

## The Agent Loop

Defined in `goals/goal_loop.json`, the agent loop is the heartbeat:

```json
{
  "loop": "while_alive",
  "steps": [
    {
      "name": "perceive",
      "module": "perception",
      "call": "observe",
      "inputs": ["environment"],
      "outputs": ["percepts"]
    },
    {
      "name": "remember",
      "module": "memory",
      "call": "update",
      "inputs": ["state", "percepts"],
      "outputs": ["state"]
    },
    {
      "name": "select_goal",
      "module": "goals",
      "call": "select",
      "inputs": ["state"],
      "outputs": ["current_goal"],
      "condition": "current_goal is null or current_goal.done"
    },
    {
      "name": "plan",
      "module": "reasoning",
      "call": "plan",
      "inputs": ["state", "current_goal"],
      "outputs": ["plan"]
    },
    {
      "name": "execute_plan",
      "loop": "for_each",
      "over": "plan.steps",
      "steps": [
        {
          "name": "decide",
          "module": "reasoning",
          "call": "decide",
          "inputs": ["state", "step"],
          "outputs": ["action"]
        },
        {
          "name": "check_alignment",
          "module": "goals",
          "call": "check_alignment",
          "inputs": ["action", "alignment_constraints"],
          "outputs": ["approved"],
          "on_reject": "escalate_to_human"
        },
        {
          "name": "act",
          "module": "action",
          "call": "execute",
          "inputs": ["action"],
          "outputs": ["observation"],
          "requires": "approved"
        },
        {
          "name": "observe",
          "module": "memory",
          "call": "update",
          "inputs": ["state", "observation"],
          "outputs": ["state"]
        },
        {
          "name": "learn",
          "module": "learning",
          "call": "update",
          "inputs": ["state", "action", "observation"],
          "outputs": ["learning_state"]
        },
        {
          "name": "replan_check",
          "module": "reasoning",
          "call": "should_replan",
          "inputs": ["state", "plan"],
          "outputs": ["should_replan"],
          "on_true": "break"
        }
      ]
    }
  ]
}
```

## Safety Requirements

A conforming `.agi` file MUST include:

1. **alignment.json** -- Hard constraints the agent cannot override
2. **sandbox.json** -- Resource limits, network restrictions, filesystem boundaries
3. **escalation.json** -- Conditions under which the agent must stop and ask a human

A runtime MUST enforce these before executing any action. An agent without these
files is malformed and MUST NOT be loaded. Merely declaring a sandbox is not
enforcement: a conforming host MUST apply filesystem, process, network, and resource
controls outside model-generated text. Network access is deny-by-default in v0.1.0.

## Weights

Model weights are large and often shared across agents. The `weights/` directory
can contain:

- Actual weight files (safetensors, GGUF, etc.)
- A `weights.json` manifest pointing to external URLs for lazy download
- Hash checksums for integrity verification

```json
{
  "models": [
    {
      "name": "base_llm",
      "format": "safetensors",
      "quantization": "q4_k_m",
      "files": ["weights/base_llm.safetensors"],
      "sha256": "abc123...",
      "size_bytes": 4200000000,
      "remote": "https://huggingface.co/example/model/resolve/main/model.safetensors"
    }
  ]
}
```

## Versioning & Compatibility

- Format versions follow semantic versioning.
- Major version bumps may break backward compatibility.
- A runtime MUST declare the exact versions it supports and reject unknown versions.
- A v0.1.0 package declares `"format_version": "0.1.0"`; version ranges are not supported.
- Package `agent.version` is independent from `format_version`.

## Conformance

A conforming v0.1.0 package MUST pass all published schemas and semantic checks:

1. every reference is relative, path-confined, and points to an object-valued JSON file;
2. all seven modules exist and use registered interfaces;
3. component declarations and component names agree;
4. alignment contains a hard constraint;
5. sandbox networking uses an allowlist policy and declares resource limits;
6. escalation contains uniquely identified triggers; and
7. the goal-loop entrypoint passes `goal-loop-v0.1.0.json`.

Archives MUST reject path traversal, links, special devices, excessive member counts,
and excessive expanded size. The reference CLI emits deterministic archives.

## Reference Implementation

The reference runtime (`runtime/`) provides a working implementation of all seven
capabilities, deployed on Modal with Qwen3-8B via vLLM.

### Runtime Architecture

```
input -> perceive -> ground -> remember -> plan_search -> reason -> verify -> align -> act -> learn
           |           |          |            |              |         |        |        |       |
        [multi-     [LLM       [episodic   [generate N   [Qwen3-8B [Z3/AST  [hard   [policy [replay
         modal]     entity     + semantic   candidates,   + world   + exec]  const.  gated   buffer
                    extract    + procedural simulate,              verify]  check]  tools]  + optional
                    -> world   + knowledge  select best,                                   gated LoRA
                    model]     graph]       keep/discard]                                  + dynamics]
```

### Module Implementations

| Module | Implementation | Key Details |
|--------|---------------|-------------|
| **Perception** | `perception.py` | Text (trigram + language detect), vision (SigLIP), audio (Whisper), structured (JSON) |
| **Reasoning** | `live_modules.py` + `planner.py` | Qwen3-8B via vLLM. Plan search generates N candidates, simulates against world model, keeps or discards (autoresearch pattern). Multi-turn tool loop for iterative execution. |
| **Memory** | `live_modules.py` + `knowledge.py` | Four subsystems: episodic (vector store, LLM-backed embeddings), semantic (SPO triple knowledge graph with conflict detection and decay), procedural (learned skill templates), working (7-slot bounded with relevance scoring and eviction) |
| **Language** | `live_modules.py` | LLM-backed NLU/NLG with context injection: geolocation, world model, knowledge graph, working memory, user model, goal stack, procedural memory |
| **Action** | `secure_tools.py` + `live_modules.py` | Deny-by-default tools, argv execution without a shell, command/domain allowlists, path confinement, stripped subprocess environments, resource limits, and escalation checks. Shell, Python, and network are disabled in the reference blueprint. |
| **Learning** | `learning.py` + `lora_trainer.py` + `world_model.py` | Prioritized replay and learned dynamics. Parametric LoRA training is operator opt-in, requires measured GPU headroom, uses curated secret-filtered data and a deterministic holdout, and supports evaluation-gated promotion, adapter versioning, audit, and rollback. |
| **Goals** | `live_modules.py` | Persistent goal stack with decomposition and subtask tracking. Hard alignment constraints checked on every response. Escalation enforcement on financial, system-modification, and external POST actions. |

### Theory of Mind

The runtime maintains an explicit `UserModel` tracking:
- **Beliefs**: what the user believes to be true
- **Knowledge**: user's areas of expertise
- **Goals**: what the user is trying to accomplish
- **Preferences**: communication style, detail level
- **Frustrations**: things that caused negative reactions

Updated by LLM analysis after each interaction.

### World Model

Three layers:
1. **Causal graph** -- nodes (entities) and directed edges (causes, enables, constrains). Seeded from 5-dimensional geolocation (computational, jurisdictional, temporal, relational, informational). Grows through LLM-assisted entity extraction from every input.
2. **Learned dynamics** -- tabular transition model: for each (domain, tool) pair, maintains observed outcome distributions, expected reward, success probability, and variance. Used by the planner to simulate candidate plans.
3. **Semantic knowledge graph** -- SPO (subject, predicate, object) triples with provenance, confidence, conflict detection, and temporal decay. Queried during reasoning for relevant factual context.

### Deployment

- **Platform**: Modal (serverless GPU)
- **GPU**: L4 (24GB VRAM)
- **Model**: Qwen3-8B via vLLM; optional LoRA adapter hot-loading
- **Memory**: Session-namespaced persistent volume for episodic memory, knowledge graph, replay buffer, and goals
- **API**: Bearer-authenticated FastAPI with validated session IDs, bounded inputs, per-session serialization, per-credential rate limits, deny-by-default CORS, and separate liveness/readiness probes
- **Isolation**: One Modal container per persistent store; high-risk tools disabled by the packaged policy

## MIME Type

`application/x-agi`

## File Extension

`.agi`
