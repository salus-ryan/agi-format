"""Learning module -- experience replay, online learning, and anti-forgetting.

Three components:
  1. ReplayBuffer    -- prioritized experience storage and sampling
  2. OnlineLearner   -- LoRA fine-tuning from episodes (when GPU available)
  3. Consolidator    -- anti-forgetting via EWC + episodic replay

These compose: the replay buffer feeds batches to the online learner,
and the consolidator constrains weight updates to prevent catastrophic
forgetting.
"""

import json
import math
import random
import time
import uuid
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger()


# ── Experience ───────────────────────────────────────────────────────────────


class Experience:
    """A single experience (state, action, observation, reward) tuple."""

    __slots__ = (
        "id", "timestamp", "state_summary", "action_tool", "action_params",
        "observation", "reward", "td_error", "novelty", "priority",
    )

    def __init__(
        self,
        state_summary: str,
        action_tool: str,
        action_params: dict,
        observation: str,
        reward: float,
        td_error: float = 0.0,
        novelty: float = 0.0,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.timestamp = time.time()
        self.state_summary = state_summary
        self.action_tool = action_tool
        self.action_params = action_params
        self.observation = observation
        self.reward = reward
        self.td_error = td_error
        self.novelty = novelty
        self.priority = self._compute_priority()

    def _compute_priority(self) -> float:
        """Priority = |TD error| + reward magnitude + novelty."""
        return abs(self.td_error) + abs(self.reward) + self.novelty + 1e-6

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "state_summary": self.state_summary,
            "action_tool": self.action_tool,
            "action_params": self.action_params,
            "observation": self.observation[:500],
            "reward": self.reward,
            "td_error": self.td_error,
            "novelty": self.novelty,
            "priority": self.priority,
        }


# ── Replay Buffer ────────────────────────────────────────────────────────────


class ReplayBuffer:
    """Prioritized experience replay buffer.

    Sampling probability is proportional to priority^alpha.
    Importance-sampling weights correct for the bias.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 100000,
        persist_path: str | None = None,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.persist_path = Path(persist_path) if persist_path else None

        self.buffer: list[Experience] = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.size = 0
        self.total_added = 0

        if self.persist_path and self.persist_path.exists():
            self._load()

    @property
    def beta(self) -> float:
        """Anneal beta from beta_start to beta_end over training."""
        frac = min(1.0, self.total_added / max(1, self.beta_anneal_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def add(self, experience: Experience) -> None:
        """Add an experience to the buffer."""
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = experience.priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        self.total_added += 1

        if self.total_added % 100 == 0:
            self._save()

    def sample(self, batch_size: int = 32) -> tuple[list[Experience], np.ndarray]:
        """Sample a batch with prioritized replay.

        Returns (experiences, importance_weights).
        """
        if self.size == 0:
            return [], np.array([])

        batch_size = min(batch_size, self.size)

        # Compute sampling probabilities
        probs = self.priorities[:self.size]
        probs = probs / probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)

        # Importance-sampling weights
        min_prob = probs.min()
        max_weight = (min_prob * self.size) ** (-self.beta)
        weights = (probs[indices] * self.size) ** (-self.beta) / max_weight

        experiences = [self.buffer[i] for i in indices]
        return experiences, weights

    def update_priorities(self, experience_ids: list[str], new_td_errors: list[float]) -> None:
        """Update priorities after learning."""
        id_to_idx = {self.buffer[i].id: i for i in range(self.size)}
        for eid, td in zip(experience_ids, new_td_errors):
            if eid in id_to_idx:
                idx = id_to_idx[eid]
                self.buffer[idx].td_error = td
                self.buffer[idx].priority = self.buffer[idx]._compute_priority()
                self.priorities[idx] = self.buffer[idx].priority ** self.alpha

    def summary(self) -> dict:
        return {
            "size": self.size,
            "capacity": self.capacity,
            "total_added": self.total_added,
            "beta": round(self.beta, 3),
            "mean_priority": float(self.priorities[:self.size].mean()) if self.size > 0 else 0,
            "mean_reward": float(np.mean([e.reward for e in self.buffer[:self.size]])) if self.size > 0 else 0,
        }

    def _save(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self.buffer[:self.size]]
        # Keep only last N to avoid huge files
        data = data[-min(10000, self.capacity):]
        with open(self.persist_path, "w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            for d in data:
                exp = Experience(
                    d["state_summary"], d["action_tool"], d["action_params"],
                    d["observation"], d["reward"], d.get("td_error", 0),
                    d.get("novelty", 0),
                )
                self.add(exp)
            log.info("replay_buffer_loaded", size=self.size)
        except Exception as e:
            log.warning("replay_buffer_load_failed", error=str(e))


# ── Online Learner ───────────────────────────────────────────────────────────


class OnlineLearner:
    """Replay-based learning: curates training data from experiences.

    This module is the bridge between the replay buffer and the LoRA trainer.
    It performs three functions:
      1. Records experiences into the prioritized replay buffer
      2. Curates training data by selecting high-value experiences
         and updating priorities based on reward prediction error
      3. Tracks learning metrics (mean reward, prediction error, etc.)

    Actual gradient updates happen in lora_trainer.py. This module decides
    WHAT to train on; lora_trainer decides HOW to train.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 8,
        update_frequency: int = 10,  # curate every N experiences
    ):
        self.replay = replay_buffer
        self.batch_size = batch_size
        self.update_frequency = update_frequency

        self.updates_performed = 0
        self.total_prediction_error = 0.0
        self.experiences_since_update = 0
        self.curated_count = 0  # total experiences curated for training

    def should_learn(self) -> bool:
        """Check if it's time for a curation step."""
        return (
            self.experiences_since_update >= self.update_frequency
            and self.replay.size >= self.batch_size
        )

    def record_experience(self, experience: Experience) -> None:
        """Add experience and potentially trigger curation."""
        self.replay.add(experience)
        self.experiences_since_update += 1

    def learn_step(self) -> dict:
        """Curate replay buffer: update priorities so the LoRA trainer
        samples the most useful experiences.

        This computes reward prediction error (TD error) for each sampled
        experience and updates the replay buffer priorities accordingly.
        High-error experiences get sampled more often during training.
        """
        if self.replay.size < self.batch_size:
            return {"skipped": True, "reason": "insufficient experiences"}

        batch, weights = self.replay.sample(self.batch_size)
        self.experiences_since_update = 0
        self.updates_performed += 1
        self.curated_count += len(batch)

        # Compute reward prediction error: how surprising is each experience?
        rewards = np.array([e.reward for e in batch])
        mean_reward = rewards.mean()
        prediction_errors = [abs(e.reward - mean_reward) for e in batch]
        mean_error = float(np.mean(prediction_errors))
        self.total_prediction_error += mean_error

        # Update priorities: high prediction error = more useful for training
        self.replay.update_priorities(
            [e.id for e in batch],
            prediction_errors,
        )

        metrics = {
            "update": self.updates_performed,
            "batch_size": len(batch),
            "mean_reward": float(mean_reward),
            "mean_prediction_error": mean_error,
            "mean_weight": float(weights.mean()),
            "curated_total": self.curated_count,
        }

        log.info("learning_curation_step", **metrics)
        return metrics

    def get_high_value_experiences(self, n: int = 50) -> list[Experience]:
        """Get the highest-priority experiences for LoRA training."""
        if self.replay.size == 0:
            return []
        batch, _ = self.replay.sample(min(n, self.replay.size))
        batch.sort(key=lambda e: e.reward, reverse=True)
        return [e for e in batch if e.reward > 0.3]

    def summary(self) -> dict:
        return {
            "updates_performed": self.updates_performed,
            "avg_prediction_error": self.total_prediction_error / max(1, self.updates_performed),
            "curated_total": self.curated_count,
            "experiences_since_update": self.experiences_since_update,
            "replay": self.replay.summary(),
        }


# ── Consolidator (Anti-Forgetting) ──────────────────────────────────────────


class Consolidator:
    """Prevents catastrophic forgetting through performance monitoring
    and replay-based training data enrichment.

    Three mechanisms:
      1. Performance monitoring: tracks per-domain reward and detects regressions.
      2. Replay enrichment: when a regression is detected, surfaces old high-reward
         experiences from the regressing domain to be re-included in training.
      3. Skill registry: tracks known skills and their importance so the LoRA
         trainer can weight training data toward regressing domains.

    This module feeds INTO the LoRA trainer, not replaces it. When it detects
    a domain regression, it increases the priority of that domain's experiences
    in the replay buffer so the next LoRA training run emphasizes them.
    """

    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay = replay_buffer

        # Skill registry: tracks learned skills and their domain
        self.skill_registry: dict[str, dict] = {}

        # Performance tracking per domain
        self.domain_performance: dict[str, list[float]] = {}

        self.consolidation_count = 0
        self.regressions_detected = 0
        self.replay_boosts_applied = 0

    def register_skill(self, skill_name: str, importance: float = 0.5,
                       domain: str = "general") -> None:
        """Register a skill the agent has learned."""
        self.skill_registry[skill_name] = {
            "importance": importance,
            "domain": domain,
            "registered": time.time(),
            "last_verified": time.time(),
            "verification_count": 0,
        }

    def record_performance(self, domain: str, score: float) -> None:
        """Record performance in a domain for regression tracking."""
        if domain not in self.domain_performance:
            self.domain_performance[domain] = []
        self.domain_performance[domain].append(score)

        # Check for regression (>5% drop from peak)
        scores = self.domain_performance[domain]
        if len(scores) >= 3:
            peak = max(scores[:-1])
            current = scores[-1]
            if peak > 0 and (peak - current) / peak > 0.05:
                self.regressions_detected += 1
                log.warning(
                    "regression_detected",
                    domain=domain,
                    peak=peak,
                    current=current,
                    drop_pct=round((peak - current) / peak * 100, 1),
                )

    def consolidate(self, replay_ratio: float = 0.3) -> dict:
        """Run a consolidation step.

        Samples old experiences, boosts priority of experiences from
        regressing domains so the LoRA trainer emphasizes them.
        """
        self.consolidation_count += 1

        # Sample old experiences for replay
        replay_size = max(1, int(self.replay.size * replay_ratio))
        replay_batch, _ = self.replay.sample(min(replay_size, 32))

        # Identify regressing domains
        regressing_domains = set()
        domains = list(self.domain_performance.keys())
        domain_status = {}
        for domain in domains:
            scores = self.domain_performance[domain]
            is_regressing = (
                len(scores) >= 3
                and max(scores[:-1]) > 0
                and (max(scores[:-1]) - scores[-1]) / max(scores[:-1]) > 0.05
            )
            if is_regressing:
                regressing_domains.add(domain)
            domain_status[domain] = {
                "latest": scores[-1] if scores else 0,
                "peak": max(scores) if scores else 0,
                "trend": "regressing" if is_regressing
                         else "improving" if len(scores) >= 2 and scores[-1] >= scores[-2]
                         else "stable",
            }

        # Boost priority of experiences from regressing domains
        if regressing_domains and replay_batch:
            boosted = 0
            for exp in replay_batch:
                # Check if this experience's domain is regressing
                exp_domain = self._infer_domain(exp)
                if exp_domain in regressing_domains and exp.reward > 0.3:
                    # Boost priority so LoRA trainer samples this more
                    self.replay.update_priorities(
                        [exp.id],
                        [exp.priority * 2.0],  # double the priority
                    )
                    boosted += 1
            if boosted:
                self.replay_boosts_applied += boosted
                log.info("consolidation_boosted_experiences",
                         domains=list(regressing_domains),
                         boosted=boosted)

        result = {
            "consolidation_step": self.consolidation_count,
            "replay_batch_size": len(replay_batch),
            "registered_skills": len(self.skill_registry),
            "monitored_domains": len(domains),
            "regressing_domains": list(regressing_domains),
            "domain_status": domain_status,
            "regressions_detected": self.regressions_detected,
            "replay_boosts_applied": self.replay_boosts_applied,
        }

        log.info("consolidation_complete", **{k: v for k, v in result.items()
                                               if k not in ("domain_status",)})
        return result

    @staticmethod
    def _infer_domain(exp: Experience) -> str:
        """Infer domain from experience state summary."""
        text = exp.state_summary.lower()
        if any(w in text for w in ["code", "python", "function", "class", "bug"]):
            return "coding"
        if any(w in text for w in ["math", "calculate", "equation"]):
            return "math"
        if any(w in text for w in ["explain", "what is", "how does"]):
            return "knowledge"
        return "general"

    def summary(self) -> dict:
        return {
            "consolidation_count": self.consolidation_count,
            "registered_skills": len(self.skill_registry),
            "monitored_domains": len(self.domain_performance),
            "regressions_detected": self.regressions_detected,
            "replay_boosts_applied": self.replay_boosts_applied,
        }
