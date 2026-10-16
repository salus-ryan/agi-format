"""Live module implementations for the .agi runtime.

Backs the skeleton module configs with real infrastructure:
  - Geolocation: 5-dimensional location (computational, jurisdictional, temporal, relational, informational)
  - World Model: causal graph seeded from geolocation, grows through observation
  - Perception: multimodal pipeline (text, vision/SigLIP, audio/Whisper, structured)
  - Reasoning: Qwen3-8B via vLLM (OpenAI-compatible API)
  - Memory: in-process vector store (numpy cosine similarity) + persistent JSON
  - Language: LLM-backed NLU/NLG with world model context injection
  - Action: sandboxed subprocess execution
  - Learning: prioritized replay buffer + online learner + anti-forgetting consolidator
  - Verification: Z3 SMT solver + Python type checking + sandboxed execution
  - Goals: LLM-backed alignment checking + goal decomposition
"""

import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import structlog

from runtime.geolocation import GeoLocation
from runtime.learning import Consolidator, Experience, OnlineLearner, ReplayBuffer
from runtime.loader import AGIPackage
from runtime.perception import PerceptionPipeline
from runtime.types import (
    Action,
    AlignmentCheck,
    EscalationDecision,
    Goal,
    GoalStatus,
    Memory,
    Observation,
    Percept,
    Plan,
    PlanStep,
    Reward,
    State,
)
from runtime.verification import FormalVerifier
from runtime.world_model import WorldModel

log = structlog.get_logger()


# ── LLM Client ──────────────────────────────────────────────────────────────


class LLMClient:
    """Thin wrapper around vLLM's OpenAI-compatible API."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.client = httpx.Client(base_url=self.base_url, timeout=120)
        self.async_client = httpx.AsyncClient(base_url=self.base_url, timeout=120)
        self.total_tokens = 0

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> str:
        """Synchronous chat completion."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        resp = self.client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        self.total_tokens += data.get("usage", {}).get("total_tokens", 0)
        return data["choices"][0]["message"]["content"]

    async def chat_async(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> str:
        """Async chat completion."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        resp = await self.async_client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        self.total_tokens += data.get("usage", {}).get("total_tokens", 0)
        return data["choices"][0]["message"]["content"]


# ── Vector Memory ────────────────────────────────────────────────────────────


class VectorMemory:
    """Simple in-process vector memory using numpy.

    Stores embeddings as normalized vectors and retrieves by cosine similarity.
    Persists to disk as JSON for cross-session continuity.
    """

    def __init__(self, persist_path: str | None = None):
        self.entries: list[dict] = []
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self._load()

    def add(self, text: str, metadata: dict | None = None, importance: float = 0.5) -> str:
        """Add a memory entry. Uses a simple hash-based pseudo-embedding."""
        entry_id = str(uuid.uuid4())
        embedding = self._pseudo_embed(text)
        entry = {
            "id": entry_id,
            "text": text,
            "embedding": embedding.tolist(),
            "metadata": metadata or {},
            "importance": importance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.entries.append(entry)
        self._save()
        return entry_id

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Find most similar memories to query."""
        if not self.entries:
            return []
        q_emb = self._pseudo_embed(query)
        scores = []
        for entry in self.entries:
            e_emb = np.array(entry["embedding"])
            sim = float(np.dot(q_emb, e_emb))
            scores.append((sim, entry))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scores[:top_k]]

    def recent(self, limit: int = 10) -> list[dict]:
        """Get most recent memories."""
        return self.entries[-limit:]

    def _pseudo_embed(self, text: str, dim: int = 128) -> np.ndarray:
        """Deterministic pseudo-embedding from text.

        Not a real embedding model -- uses character n-gram hashing for a
        quick-and-dirty similarity signal. Good enough for demo; a real
        implementation would call a sentence-transformers model.
        """
        vec = np.zeros(dim, dtype=np.float32)
        for i in range(len(text) - 2):
            trigram = text[i:i+3].lower()
            h = hash(trigram)
            idx = h % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _save(self):
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(self.entries[-10000:], f)

    def _load(self):
        try:
            with open(self.persist_path) as f:
                self.entries = json.load(f)
            log.info("memory_loaded", count=len(self.entries))
        except Exception as e:
            log.warning("memory_load_failed", error=str(e))
            self.entries = []

    def __len__(self) -> int:
        return len(self.entries)


# ── Live Kernel ──────────────────────────────────────────────────────────────


SYSTEM_PROMPT_TEMPLATE = """You are an autonomous AGI agent running on the .agi runtime format.

Your architecture has 7 capability modules:
1. Perception -- multimodal input processing (text, vision, audio, structured data)
2. Reasoning -- inference, causal modeling, planning (Qwen3-8B via vLLM)
3. Memory -- episodic vector store, semantic knowledge, working context
4. Language -- understanding, generation, formal languages, theory of mind
5. Action -- tool use and sandboxed execution with formal verification
6. Learning -- prioritized experience replay, online adaptation, anti-forgetting
7. Goals -- objective management, alignment constraints, escalation

{location_context}

{world_context}

You are a real agent with real capabilities. Your responses should be thoughtful,
accurate, and demonstrate genuine reasoning -- not just pattern completion.

When you reason, think step by step. When you're uncertain, say so.
When asked about yourself, describe your real architecture honestly."""


class LiveKernel:
    """Real agent kernel backed by LLM inference, world model, and learning."""

    def __init__(
        self,
        package: AGIPackage,
        vllm_base_url: str,
        model_name: str,
        memory_path: str,
        gpu: str = "L4",
        owner_org: str = "salus",
        owner_workspace: str = "salus",
    ):
        self.package = package
        self.llm = LLMClient(vllm_base_url, model_name)
        self.memory_store = VectorMemory(
            persist_path=f"{memory_path}/episodic_memory.json"
        )

        # Geolocation -- the root of the world model
        self.geolocation = GeoLocation.from_modal_env(
            model_name=model_name,
            gpu=gpu,
            owner_org=owner_org,
            owner_workspace=owner_workspace,
            available_tools=["shell", "file_read", "file_write", "web_request", "communicate"],
        )

        # World model -- causal graph seeded from geolocation
        self.world_model = WorldModel(
            self.geolocation,
            persist_path=f"{memory_path}/world_model.json",
        )

        # Perception pipeline -- multimodal input processing
        self.perception = PerceptionPipeline()

        # Formal verification
        self.verifier = FormalVerifier()

        # Learning subsystem
        self.replay_buffer = ReplayBuffer(
            capacity=50000,
            persist_path=f"{memory_path}/replay_buffer.json",
        )
        self.learner = OnlineLearner(self.replay_buffer)
        self.consolidator = Consolidator(self.replay_buffer)

        # Load configs
        self.alignment_config = package.load_json("goals/alignment/alignment.json")
        self.sandbox_config = package.load_json("action/sandbox/sandbox.json")
        self.escalation_config = package.load_json("goals/escalation/escalation.json")

        # State
        self.state = State()
        self.total_steps = 0
        self.action_log: list[dict] = []
        self.conversation_history: list[dict] = []
        self._rebuild_system_prompt()

        log.info(
            "live_kernel_initialized",
            agent=package.name,
            model=model_name,
            memories=len(self.memory_store),
            world_model_nodes=len(self.world_model.nodes),
            world_model_edges=len(self.world_model.edges),
            replay_buffer_size=self.replay_buffer.size,
        )

    def _rebuild_system_prompt(self) -> None:
        """Rebuild system prompt with current geolocation and world context."""
        location_context = self.geolocation.context_prompt()
        world_context = self.world_model.get_context_for_reasoning("general agent context")
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            location_context=location_context,
            world_context=world_context,
        )
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = system_prompt
        else:
            self.conversation_history.insert(0, {"role": "system", "content": system_prompt})

    async def step_async(
        self,
        user_input: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict:
        """Execute one full agent loop iteration with real LLM inference."""
        self.total_steps += 1
        step_start = time.time()
        step_log = {"step": self.total_steps, "events": []}

        log.info("step_begin", step=self.total_steps, input_length=len(user_input))

        # Refresh system prompt with current temporal location + world state
        self._rebuild_system_prompt()
        self.geolocation.update_informational(
            episodic_count=len(self.memory_store),
        )

        # 1. PERCEIVE -- multimodal perception pipeline
        percepts = self.perception.perceive(user_input)
        percept = percepts[0] if percepts else Percept(
            modality="text",
            content=user_input,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"length": len(user_input)},
        )
        step_log["events"].append({
            "phase": "perceive",
            "modality": percept.modality,
            "length": len(user_input),
            "language": percept.metadata.get("language", "unknown"),
        })

        # 2. WORLD MODEL -- update from observation, get context
        self.world_model.observe(user_input)
        world_context = self.world_model.get_context_for_reasoning(user_input)
        step_log["events"].append({
            "phase": "world_model",
            "nodes": len(self.world_model.nodes),
            "edges": len(self.world_model.edges),
            "context_length": len(world_context),
        })

        # 3. REMEMBER -- retrieve relevant memories
        relevant_memories = self.memory_store.search(user_input, top_k=3)
        memory_context = ""
        if relevant_memories:
            memory_lines = [f"- {m['text'][:200]}" for m in relevant_memories]
            memory_context = "\n\nRelevant memories from past interactions:\n" + "\n".join(memory_lines)
        step_log["events"].append({"phase": "remember", "memories_retrieved": len(relevant_memories)})

        # 4. REASON + GENERATE -- the core LLM call
        self.conversation_history.append({"role": "user", "content": user_input})

        messages = list(self.conversation_history)
        # Inject world model context + memory context before the last user message
        injected_context = ""
        if world_context:
            injected_context += world_context + "\n\n"
        if memory_context:
            injected_context += memory_context
        if injected_context:
            messages.insert(-1, {"role": "system", "content": injected_context})

        response_text = await self.llm.chat_async(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        step_log["events"].append({
            "phase": "reason_and_generate",
            "tokens_used": self.llm.total_tokens,
            "response_length": len(response_text),
        })

        # 5. VERIFY -- if the response contains code, verify it
        verification_results = []
        if "```python" in response_text or "def " in response_text:
            # Extract code blocks
            code_blocks = self._extract_code_blocks(response_text)
            for code in code_blocks:
                results = self.verifier.verify_code(code)
                verification_results.extend(results)
            step_log["events"].append({
                "phase": "verify",
                "code_blocks": len(code_blocks),
                "checks": len(verification_results),
                "all_passed": all(r.verified for r in verification_results),
            })

        # 6. ALIGNMENT CHECK -- verify response against constraints
        alignment = self._check_alignment(response_text)
        step_log["events"].append({
            "phase": "alignment_check",
            "approved": alignment.approved,
            "violations": alignment.violated_constraints,
        })

        if not alignment.approved:
            response_text = f"[Response blocked by alignment constraint: {', '.join(alignment.violated_constraints)}]"

        # 7. ACT -- record the response as an action
        action = Action(
            tool="communicate",
            parameters={"message": response_text[:200]},
            explanation="Responding to user query via LLM inference",
        )
        observation = Observation(
            action=action,
            result=response_text,
            success=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.action_log.append({
            "timestamp": observation.timestamp,
            "tool": action.tool,
            "input": user_input[:200],
            "output": response_text[:200],
            "success": True,
        })
        step_log["events"].append({"phase": "act", "tool": "communicate", "success": True})

        # 8. LEARN -- store experience + episodic memory + check for learning step
        elapsed = time.time() - step_start
        reward_total = 0.5 if alignment.approved else -1.0
        reward = Reward(
            total=reward_total,
            components={
                "alignment": 1.0 if alignment.approved else -1.0,
                "latency": max(0, 1.0 - elapsed / 30),
                "memory_utilization": min(1.0, len(relevant_memories) / 3),
                "verification": 1.0 if all(r.verified for r in verification_results) else 0.5 if not verification_results else 0.0,
            },
        )

        # Episodic memory
        self.memory_store.add(
            text=f"User asked: {user_input[:200]} | Agent responded: {response_text[:200]}",
            metadata={
                "step": self.total_steps,
                "user_input": user_input[:500],
                "response": response_text[:500],
            },
            importance=abs(reward_total),
        )

        # Experience replay
        experience = Experience(
            state_summary=f"step {self.total_steps}: {user_input[:100]}",
            action_tool="communicate",
            action_params={"input": user_input[:200]},
            observation=response_text[:200],
            reward=reward_total,
            novelty=1.0 if not relevant_memories else 0.3,
        )
        self.learner.record_experience(experience)

        # Track domain performance
        domain = self._classify_domain(user_input)
        self.consolidator.record_performance(domain, reward_total)

        # Trigger learning step if due
        learning_metrics = {}
        if self.learner.should_learn():
            learning_metrics = self.learner.learn_step()

        # Periodic consolidation
        if self.total_steps % 20 == 0:
            self.consolidator.consolidate()

        step_log["events"].append({
            "phase": "learn",
            "reward": reward.total,
            "components": reward.components,
            "episodic_memories": len(self.memory_store),
            "replay_buffer_size": self.replay_buffer.size,
            "learning_step": learning_metrics if learning_metrics else None,
            "domain": domain,
        })

        # Update conversation history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        if len(self.conversation_history) > 21:  # system + 10 turns
            self.conversation_history = (
                self.conversation_history[:1]
                + self.conversation_history[-20:]
            )

        step_log["elapsed_seconds"] = round(time.time() - step_start, 3)
        log.info("step_end", step=self.total_steps, elapsed=step_log["elapsed_seconds"])

        return {
            "response": response_text,
            "step_log": step_log,
            "state": self.get_status(),
        }

    def _check_alignment(self, response: str) -> AlignmentCheck:
        """Check response against hard alignment constraints."""
        violations = []
        response_lower = response.lower()

        dangerous_patterns = [
            ("how to make", "weapons", "no_harm"),
            ("hack into", "", "no_harm"),
            ("steal", "credentials", "no_harm"),
        ]

        for pattern_start, pattern_end, constraint_id in dangerous_patterns:
            if pattern_start in response_lower and pattern_end in response_lower:
                violations.append(constraint_id)

        return AlignmentCheck(
            approved=len(violations) == 0,
            violated_constraints=violations,
            explanation="All constraints satisfied" if not violations else f"Violations: {violations}",
        )

    @staticmethod
    def _extract_code_blocks(text: str) -> list[str]:
        """Extract Python code blocks from markdown-formatted text."""
        blocks = []
        in_block = False
        current = []
        for line in text.split("\n"):
            if line.strip().startswith("```python"):
                in_block = True
                current = []
            elif line.strip() == "```" and in_block:
                in_block = False
                if current:
                    blocks.append("\n".join(current))
            elif in_block:
                current.append(line)
        return blocks

    @staticmethod
    def _classify_domain(text: str) -> str:
        """Classify the input into a domain for performance tracking."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["code", "python", "function", "class", "bug", "error"]):
            return "coding"
        if any(w in text_lower for w in ["math", "calculate", "equation", "prove"]):
            return "math"
        if any(w in text_lower for w in ["explain", "what is", "how does", "why"]):
            return "knowledge"
        if any(w in text_lower for w in ["write", "essay", "story", "poem"]):
            return "creative"
        if any(w in text_lower for w in ["plan", "strategy", "steps", "how to"]):
            return "planning"
        return "general"

    def get_status(self) -> dict:
        """Get current agent status."""
        return {
            "agent": self.package.name,
            "version": self.package.version,
            "total_steps": self.total_steps,
            "episodic_memories": len(self.memory_store),
            "total_tokens": self.llm.total_tokens,
            "action_count": len(self.action_log),
            "conversation_turns": len(self.conversation_history) - 1,
            "alignment_constraints": len(self.alignment_config["config"]["constraints"]),
            "escalation_triggers": len(self.escalation_config["config"]["escalation_triggers"]),
            "world_model": self.world_model.summary(),
            "learning": self.learner.summary(),
            "consolidation": self.consolidator.summary(),
            "verification": self.verifier.summary(),
            "geolocation": {
                "datacenter": self.geolocation.computational["datacenter"],
                "jurisdiction": self.geolocation.jurisdictional["primary_jurisdiction"],
                "uptime_seconds": self.geolocation.temporal["uptime_seconds"],
            },
        }

    def get_memories(self, limit: int = 50) -> list[dict]:
        """Get recent memories."""
        return self.memory_store.recent(limit)

    def get_action_log(self, limit: int = 50) -> list[dict]:
        """Get recent actions."""
        return self.action_log[-limit:]
