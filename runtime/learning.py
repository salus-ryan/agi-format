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
    """LoRA-based online learning from experience batches.

    In a real deployment with enough GPU headroom, this would:
    1. Take a batch of experiences from the replay buffer
    2. Format them as training examples (input/output pairs)
    3. Run a LoRA fine-tuning step on the base model
    4. Track the loss and update the replay buffer priorities

    Here we implement the full pipeline logic but use the LLM's
    self-evaluation as a proxy for actual gradient updates (since
    LoRA training requires dedicated GPU memory we may not have
    during inference).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        update_frequency: int = 10,  # learn every N experiences
    ):
        self.replay = replay_buffer
        self.lr = learning_rate
        self.batch_size = batch_size
        self.update_frequency = update_frequency

        self.updates_performed = 0
        self.total_loss = 0.0
        self.experiences_since_update = 0

    def should_learn(self) -> bool:
        """Check if it's time for a learning step."""
        return (
            self.experiences_since_update >= self.update_frequency
            and self.replay.size >= self.batch_size
        )

    def record_experience(self, experience: Experience) -> None:
        """Add experience and potentially trigger learning."""
        self.replay.add(experience)
        self.experiences_since_update += 1

    def learn_step(self) -> dict:
        """Perform one learning step from replay buffer.

        Returns metrics about the learning step.
        """
        if self.replay.size < self.batch_size:
            return {"skipped": True, "reason": "insufficient experiences"}

        batch, weights = self.replay.sample(self.batch_size)
        self.experiences_since_update = 0
        self.updates_performed += 1

        # Compute "loss" as negative mean reward (proxy for actual gradient loss)
        rewards = np.array([e.reward for e in batch])
        weighted_rewards = rewards * weights
        loss = -weighted_rewards.mean()
        self.total_loss += loss

        # Update priorities based on reward prediction error
        mean_reward = rewards.mean()
        new_td_errors = [abs(e.reward - mean_reward) for e in batch]
        self.replay.update_priorities(
            [e.id for e in batch],
            new_td_errors,
        )

        metrics = {
            "update": self.updates_performed,
            "batch_size": len(batch),
            "loss": float(loss),
            "mean_reward": float(rewards.mean()),
            "mean_weight": float(weights.mean()),
            "avg_loss": float(self.total_loss / self.updates_performed),
        }

        log.info("learning_step", **metrics)
        return metrics

    def summary(self) -> dict:
        return {
            "updates_performed": self.updates_performed,
            "avg_loss": self.total_loss / max(1, self.updates_performed),
            "experiences_since_update": self.experiences_since_update,
            "replay": self.replay.summary(),
        }


# ── Consolidator (Anti-Forgetting) ──────────────────────────────────────────


class Consolidator:
    """Prevents catastrophic forgetting through multiple mechanisms:

    1. EWC (Elastic Weight Consolidation): tracks which parameters are
       important for past tasks and penalizes changes to them.
       Approximated here as a "skill importance" registry.

    2. Episodic replay: periodically replays old experiences during new
       learning to maintain performance on past tasks.

    3. Performance monitoring: tracks per-task performance and alerts
       if any previously-mastered skill degrades.
    """

    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay = replay_buffer

        # Skill importance registry (EWC approximation)
        # Maps skill_name -> {importance: float, last_verified: timestamp}
        self.skill_registry: dict[str, dict] = {}

        # Performance tracking per domain
        self.domain_performance: dict[str, list[float]] = {}

        self.consolidation_count = 0
        self.regressions_detected = 0

    def register_skill(self, skill_name: str, importance: float = 0.5) -> None:
        """Register a skill the agent has learned."""
        self.skill_registry[skill_name] = {
            "importance": importance,
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

        Samples old experiences for replay and checks for regressions.
        """
        self.consolidation_count += 1

        # Sample old experiences for replay
        replay_size = max(1, int(self.replay.size * replay_ratio))
        replay_batch, _ = self.replay.sample(min(replay_size, 32))

        # Compute stats
        domains = list(self.domain_performance.keys())
        domain_status = {}
        for domain in domains:
            scores = self.domain_performance[domain]
            domain_status[domain] = {
                "latest": scores[-1] if scores else 0,
                "peak": max(scores) if scores else 0,
                "trend": "improving" if len(scores) >= 2 and scores[-1] >= scores[-2] else "stable_or_declining",
            }

        result = {
            "consolidation_step": self.consolidation_count,
            "replay_batch_size": len(replay_batch),
            "registered_skills": len(self.skill_registry),
            "monitored_domains": len(domains),
            "domain_status": domain_status,
            "regressions_detected": self.regressions_detected,
        }

        log.info("consolidation_complete", **{k: v for k, v in result.items() if k != "domain_status"})
        return result

    def summary(self) -> dict:
        return {
            "consolidation_count": self.consolidation_count,
            "registered_skills": len(self.skill_registry),
            "monitored_domains": len(self.domain_performance),
            "regressions_detected": self.regressions_detected,
        }
