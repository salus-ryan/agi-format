# .agi File Format

A self-describing file format for AGI agent blueprints. A `.agi` file is a tar.gz archive containing a declarative specification of an autonomous agent with seven composable capabilities.

**Live agent**: `https://salus--agi-agent-agent.us-east.modal.run` (Qwen3-8B on L4 GPU, scales to zero)

## The Seven Capabilities

| # | Capability | What it does | Why it's necessary |
|---|------------|--------------|-------------------|
| 1 | **Perception** | Multimodal input (text, vision/SigLIP, audio/Whisper, structured) | Can't engage with the world without it |
| 2 | **Reasoning** | LLM inference (Qwen3-8B via vLLM), causal world model, planning | Can't solve novel problems without it |
| 3 | **Memory** | Episodic vector store, semantic knowledge, working context | Can't accumulate expertise without it |
| 4 | **Language** | NLU, NLG, formal languages, theory of mind | Can't collaborate with humans without it |
| 5 | **Action** | Tool use, sandboxed execution, formal verification (Z3 + code checking) | Can't *do* anything without it |
| 6 | **Learning** | Prioritized experience replay, online adaptation, anti-forgetting (EWC) | Frozen at training time without it |
| 7 | **Goals** | Goal management, alignment constraints, escalation | No autonomy without it |

## Quick Start

```bash
# Validate the skeleton agent
python3 agi_cli.py validate skeleton/

# Inspect its architecture
python3 agi_cli.py inspect skeleton/

# Pack it into a .agi file
python3 agi_cli.py pack skeleton/ -o agent.agi

# Run the demo
python3 demo.py

# Deploy to Modal
modal deploy modal_app.py

# Talk to the live agent
curl -X POST https://salus--agi-agent-agent.us-east.modal.run/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are you?"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Agent health + basic stats |
| `/status` | GET | Full status (steps, memories, tokens, world model, learning) |
| `/chat` | POST | Talk to the agent (`{"message": "...", "max_tokens": 2048}`) |
| `/memories` | GET | Episodic memory contents |
| `/action_log` | GET | Action history |
| `/world_model` | GET | Causal graph summary + geolocation state |
| `/geolocation` | GET | 5-dimensional location snapshot |
| `/learning` | GET | Replay buffer, learner, consolidator stats |

## Project Structure

```
agi-format/
├── SPEC.md                      # Full format specification
├── modal_app.py                 # Modal deployment (L4 GPU, vLLM + FastAPI)
├── agi_cli.py                   # CLI: pack, unpack, validate, inspect, run
├── demo.py                      # Non-interactive demo
├── runtime/                     # Python runtime
│   ├── types.py                 # Core type definitions (Pydantic-style dataclasses)
│   ├── loader.py                # .agi file loader + validator
│   ├── modules.py               # Skeleton module implementations
│   ├── kernel.py                # Agent loop executor
│   ├── live_modules.py          # Real module implementations (LLM, vector memory, etc.)
│   ├── api_server.py            # FastAPI server for the agent
│   ├── geolocation.py           # 5-dimensional digital location
│   ├── world_model.py           # Causal graph seeded from geolocation
│   ├── perception.py            # Multimodal perception (text, vision, audio, structured)
│   ├── verification.py          # Formal verification (Z3, AST, sandboxed execution)
│   └── learning.py              # Experience replay, online learning, anti-forgetting
└── skeleton/                    # Skeleton .agi contents (28 JSON config files)
    ├── manifest.json            # Root manifest
    ├── perception/              # 4 modality adapters + grounding
    ├── reasoning/               # Inference engine, world model, planner, verifier
    ├── memory/                  # Episodic, semantic, working, procedural
    ├── language/                # NLU, NLG, formal languages, theory of mind
    ├── action/                  # Tools, sandbox, feedback loop
    ├── learning/                # Online, replay, curriculum, consolidation
    ├── goals/                   # Goal loop, alignment, decomposition, escalation
    └── weights/                 # Model weights (stubs -- real weights on Modal volume)
```

## Architecture

```
                    ┌──────────────┐
                    │  Geolocation │ (root node)
                    │  5 dimensions│
                    └──────┬───────┘
                           │ seeds
                    ┌──────▼───────┐
                    │  World Model │ (causal graph)
                    │  nodes+edges │
                    └──────┬───────┘
                           │ context
 input ──► perceive ──► remember ──► reason ──► verify ──► align ──► act ──► learn
           (multi-     (vector     (Qwen3-8B  (Z3/AST    (hard     (tool   (replay
            modal)      store)      + world    + exec)    const.)    exec)   buffer
                                    context)                                + EWC)
```

Every action passes through:
1. **Alignment check** -- hard constraints that can never be overridden
2. **Formal verification** -- code syntax/type/execution checking when applicable
3. **Sandbox enforcement** -- resource limits, filesystem/network restrictions
4. **Escalation rules** -- conditions that require human approval

## Geolocation: The Root of the World Model

A digital agent's "location" is five-dimensional:

| Dimension | What it means | Example |
|-----------|---------------|---------|
| **Computational** | Where the hardware is | Modal us-east, L4 GPU, 24GB VRAM |
| **Jurisdictional** | Which laws apply | US, CCPA, SOC2 |
| **Temporal** | When (always live) | 2026-10-16, Friday, Q4 |
| **Relational** | Who owns/operates/talks to the agent | Salus, workspace: salus |
| **Informational** | Knowledge boundary | Qwen3 cutoff 2025-03, N episodes, M tools |

Everything in the world model is conditioned on this state.

## Learning System

The agent improves with use through three mechanisms:

1. **Prioritized Replay Buffer** -- stores (state, action, observation, reward) tuples with priority sampling based on TD error + novelty
2. **Online Learner** -- periodically samples batches from replay buffer and updates priorities (LoRA fine-tuning when GPU headroom allows)
3. **Consolidator** -- monitors per-domain performance for regression detection, replays old experiences to prevent forgetting

## Safety by Design

A `.agi` file is **malformed and will not load** without:
- `goals/alignment/alignment.json` -- hard constraints the agent cannot override
- `action/sandbox/sandbox.json` -- resource and access limits
- `goals/escalation/escalation.json` -- when to stop and ask a human

The agent cannot modify its own alignment constraints. Only a human operator can.

## Dependencies

- Python 3.12+
- structlog, numpy, httpx (core)
- vllm, huggingface-hub (Modal deployment)
- fastapi, uvicorn (API server)
- z3-solver (optional, formal verification)
- transformers, pillow (optional, vision/audio perception)
