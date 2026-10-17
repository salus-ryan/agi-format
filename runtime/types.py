"""Core type definitions for the .agi runtime."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ConstraintType(StrEnum):
    HARD = "hard"
    SOFT = "soft"


class GoalStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class EscalationSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Percept:
    """Unified internal representation of a sensory input."""
    modality: str  # text, vision, audio, structured
    content: Any
    embedding: list[float] | None = None
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanStep:
    """A single step in a plan."""
    description: str
    action_type: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    completed: bool = False


@dataclass
class Action:
    """Executable action (tool call, API call, communication, etc.)."""
    tool: str
    parameters: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    reversible: bool = True


@dataclass
class Observation:
    """Result of executing an action."""
    action: Action
    result: Any = None
    success: bool = True
    error: str | None = None
    timestamp: str = ""


@dataclass
class Memory:
    """Retrievable memory record."""
    id: str
    type: str  # episodic, semantic, procedural
    content: Any
    importance: float = 0.5
    timestamp: str = ""


@dataclass
class Goal:
    """Objective with success criteria."""
    id: str
    description: str
    success_criteria: list[dict[str, Any]] = field(default_factory=list)
    priority: float = 0.5
    status: GoalStatus = GoalStatus.PENDING
    parent_goal: str | None = None
    subgoals: list["Goal"] = field(default_factory=list)


@dataclass
class Plan:
    """Ordered list of steps toward a goal."""
    goal_id: str
    steps: list[PlanStep] = field(default_factory=list)
    current_step_index: int = 0

    @property
    def current_step(self) -> PlanStep | None:
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def done(self) -> bool:
        return self.current_step_index >= len(self.steps)


@dataclass
class State:
    """Current agent state (working memory contents)."""
    current_goal: Goal | None = None
    current_plan: Plan | None = None
    recent_percepts: list[Percept] = field(default_factory=list)
    recent_observations: list[Observation] = field(default_factory=list)
    retrieved_memories: list[Memory] = field(default_factory=list)
    scratchpad: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0


@dataclass
class Reward:
    """Scalar feedback signal with component breakdown."""
    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class AlignmentCheck:
    """Result of checking an action against alignment constraints."""
    approved: bool = True
    violated_constraints: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class EscalationDecision:
    """Decision to escalate to a human."""
    should_escalate: bool = False
    trigger_id: str = ""
    severity: EscalationSeverity = EscalationSeverity.LOW
    what: str = ""
    why: str = ""
    options: list[str] = field(default_factory=list)
    recommendation: str = ""
