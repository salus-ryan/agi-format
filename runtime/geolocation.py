"""Geolocation module -- the root node of the world model.

A digital agent's "location" is five-dimensional:
  1. Computational  -- where the hardware is, what it can do
  2. Jurisdictional -- which laws and regulations apply
  3. Temporal       -- when, what's true about the world right now
  4. Relational     -- who owns/operates the agent, who it's talking to
  5. Informational  -- what the agent has access to, its knowledge boundary

Everything in the world model is conditioned on this state.
"""

import os
import platform
import time
from datetime import datetime, timezone

import structlog

log = structlog.get_logger()


class GeoLocation:
    """Five-dimensional location state for a digital agent."""

    def __init__(
        self,
        # Computational
        datacenter: str = "unknown",
        region: str = "unknown",
        gpu: str = "none",
        gpu_vram_gb: float = 0,
        cpu_count: int = 0,
        memory_gb: float = 0,
        # Jurisdictional
        jurisdiction: str = "US",
        compliance_frameworks: list[str] | None = None,
        # Relational
        owner_org: str = "unknown",
        owner_workspace: str = "unknown",
        operator: str = "unknown",
        trust_boundary: str = "authenticated",
        # Informational
        model_name: str = "unknown",
        model_training_cutoff: str = "unknown",
        available_tools: list[str] | None = None,
        knowledge_sources: list[str] | None = None,
    ):
        # 1. Computational location
        self.computational = {
            "datacenter": datacenter,
            "region": region,
            "gpu": gpu,
            "gpu_vram_gb": gpu_vram_gb,
            "cpu_count": cpu_count or os.cpu_count() or 0,
            "memory_gb": memory_gb,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }

        # 2. Jurisdictional location
        self.jurisdictional = {
            "primary_jurisdiction": jurisdiction,
            "compliance_frameworks": compliance_frameworks or [],
            "data_residency": region,
            "applicable_regulations": self._infer_regulations(jurisdiction),
        }

        # 3. Temporal location (always live)
        self._boot_time = time.time()

        # 4. Relational location
        self.relational = {
            "owner_org": owner_org,
            "owner_workspace": owner_workspace,
            "operator": operator,
            "trust_boundary": trust_boundary,
            "current_interlocutor": None,
            "session_id": None,
        }

        # 5. Informational location
        self.informational = {
            "model_name": model_name,
            "model_training_cutoff": model_training_cutoff,
            "available_tools": available_tools or [],
            "knowledge_sources": knowledge_sources or [],
            "episodic_memory_count": 0,
            "semantic_facts_count": 0,
            "procedural_skills_count": 0,
        }

        log.info(
            "geolocation_initialized",
            datacenter=datacenter,
            region=region,
            gpu=gpu,
            jurisdiction=jurisdiction,
            owner=owner_org,
            model=model_name,
        )

    @property
    def temporal(self) -> dict:
        """Temporal location is always computed live."""
        now = datetime.now(timezone.utc)
        return {
            "utc_now": now.isoformat(),
            "unix_timestamp": time.time(),
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "year": now.year,
            "quarter": f"Q{(now.month - 1) // 3 + 1}",
            "uptime_seconds": round(time.time() - self._boot_time, 1),
        }

    def set_interlocutor(self, identity: str, session_id: str | None = None) -> None:
        """Update who the agent is currently talking to."""
        self.relational["current_interlocutor"] = identity
        self.relational["session_id"] = session_id
        log.info("interlocutor_set", identity=identity, session_id=session_id)

    def update_informational(
        self,
        episodic_count: int | None = None,
        semantic_count: int | None = None,
        procedural_count: int | None = None,
    ) -> None:
        """Update the agent's knowledge boundary counts."""
        if episodic_count is not None:
            self.informational["episodic_memory_count"] = episodic_count
        if semantic_count is not None:
            self.informational["semantic_facts_count"] = semantic_count
        if procedural_count is not None:
            self.informational["procedural_skills_count"] = procedural_count

    def snapshot(self) -> dict:
        """Full five-dimensional location snapshot."""
        return {
            "computational": self.computational,
            "jurisdictional": self.jurisdictional,
            "temporal": self.temporal,
            "relational": self.relational,
            "informational": self.informational,
        }

    def context_prompt(self) -> str:
        """Generate a natural-language context string for the LLM.

        This gets injected into the system prompt so the model knows
        where/when/who it is -- grounding every inference in reality.
        """
        t = self.temporal
        c = self.computational
        j = self.jurisdictional
        r = self.relational
        i = self.informational

        lines = [
            "## Agent Location Context (auto-generated, always current)",
            "",
            f"**When**: {t['date']} ({t['day_of_week']}), {t['utc_now']}",
            f"**Where**: {c['datacenter']} ({c['region']}), {c['gpu']} GPU, {c['gpu_vram_gb']}GB VRAM",
            f"**Jurisdiction**: {j['primary_jurisdiction']}",
        ]
        if j["applicable_regulations"]:
            lines.append(f"**Regulations**: {', '.join(j['applicable_regulations'])}")
        lines.append(f"**Owner**: {r['owner_org']} (workspace: {r['owner_workspace']})")
        if r["current_interlocutor"]:
            lines.append(f"**Talking to**: {r['current_interlocutor']}")
        lines.append(f"**Model**: {i['model_name']} (training cutoff: {i['model_training_cutoff']})")
        lines.append(f"**Knowledge**: {i['episodic_memory_count']} episodes, "
                      f"{i['semantic_facts_count']} facts, "
                      f"{i['procedural_skills_count']} procedures")
        if i["available_tools"]:
            lines.append(f"**Tools**: {', '.join(i['available_tools'])}")
        lines.append(f"**Uptime**: {t['uptime_seconds']}s")
        return "\n".join(lines)

    @staticmethod
    def _infer_regulations(jurisdiction: str) -> list[str]:
        """Infer applicable regulations from jurisdiction."""
        regulations = {
            "US": ["CCPA", "FCRA", "ECOA", "GLBA", "SEC regulations"],
            "EU": ["GDPR", "AI Act", "ePrivacy Directive"],
            "UK": ["UK GDPR", "Data Protection Act 2018"],
            "CA": ["PIPEDA", "AIDA"],
        }
        return regulations.get(jurisdiction, [])

    @classmethod
    def from_modal_env(
        cls,
        model_name: str = "unknown",
        gpu: str = "unknown",
        owner_org: str = "unknown",
        owner_workspace: str = "unknown",
        available_tools: list[str] | None = None,
    ) -> "GeoLocation":
        """Factory: build geolocation from Modal runtime environment."""
        region = os.environ.get("MODAL_REGION", "us-east")
        # Modal doesn't expose datacenter directly, but we know it's Modal
        datacenter = f"modal-{region}"

        gpu_vram = {
            "T4": 16, "L4": 24, "A10G": 24, "L40S": 48,
            "A100": 80, "A100-80GB": 80, "H100": 80, "H200": 141,
        }

        return cls(
            datacenter=datacenter,
            region=region,
            gpu=gpu,
            gpu_vram_gb=gpu_vram.get(gpu, 0),
            jurisdiction="US",  # Modal US regions
            compliance_frameworks=["SOC2"],
            owner_org=owner_org,
            owner_workspace=owner_workspace,
            operator="modal",
            model_name=model_name,
            model_training_cutoff="2025-03",  # Qwen3 cutoff
            available_tools=available_tools or [
                "shell", "file_read", "file_write", "web_request", "communicate",
            ],
            knowledge_sources=["episodic_memory", "conversation_history"],
        )
