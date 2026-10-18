"""Deterministic evaluation and enforcement of an `.agi` boundary policy.

Every decision is made from the package policy and the proposed call alone. No model
output, keyword heuristics, or network lookups participate in allow/deny decisions.
"""

from __future__ import annotations

import json
import re
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog

from runtime.secure_tools import ToolExecutor

log = structlog.get_logger()

BUILTIN_TOOLS = frozenset({"file_read", "file_write", "shell", "python_eval", "web_request"})
MAX_MATCHED_PARAM_CHARS = 4096
MAX_AUDIT_ENTRIES = 1000


class PolicyError(ValueError):
    """Raised when a policy is internally inconsistent."""


class Verdict(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Decision:
    """Outcome of evaluating one proposed tool call against the policy."""

    verdict: Verdict
    tool: str
    reason: str
    scope: str | None = None
    trigger_id: str | None = None
    severity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "tool": self.tool,
            "reason": self.reason,
            "scope": self.scope,
            "trigger_id": self.trigger_id,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ToolRule:
    name: str
    scope: str
    requires_approval: bool = False


@dataclass(frozen=True)
class Trigger:
    id: str
    description: str
    severity: str
    tools: frozenset[str]
    params: tuple[tuple[str, re.Pattern[str]], ...] = ()

    def matches(self, tool: str, params: Mapping[str, Any]) -> bool:
        if "*" not in self.tools and tool not in self.tools:
            return False
        for name, pattern in self.params:
            if name not in params:
                return False
            if not pattern.search(_param_text(params[name])):
                return False
        return True


def _param_text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return text[:MAX_MATCHED_PARAM_CHARS]


@dataclass(frozen=True)
class Policy:
    """A validated, immutable boundary policy."""

    tools: Mapping[str, ToolRule]
    triggers: tuple[Trigger, ...]
    sandbox: Mapping[str, Any]
    max_actions_per_session: int | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Policy:
        """Build a policy from schema-valid JSON, enforcing cross-field rules."""
        tools: dict[str, ToolRule] = {}
        for item in data["tools"]:
            name = item["name"]
            if name in tools:
                raise PolicyError(f"duplicate tool {name!r}")
            tools[name] = ToolRule(
                name=name,
                scope=item.get("scope", f"tool:{name}"),
                requires_approval=bool(item.get("requires_approval", False)),
            )

        sandbox = data["sandbox"]
        execution = sandbox["execution"]
        isolation = sandbox["isolation"]
        network = sandbox["network"]
        process_enabled = execution["shell_enabled"] or execution["python_enabled"]
        if isolation["type"] == "disabled" and process_enabled:
            raise PolicyError("disabled process isolation cannot enable shell or Python")
        if isolation["type"] == "oci":
            if "runtime" not in isolation:
                raise PolicyError("OCI isolation must declare a runtime")
            if not _IMAGE_BY_DIGEST.fullmatch(isolation.get("image", "")):
                raise PolicyError("OCI isolation image must be pinned by sha256 digest")
        if "shell" in tools and not execution["shell_enabled"]:
            raise PolicyError("tool 'shell' is declared but execution.shell_enabled is false")
        if "shell" in tools and not execution["allowed_commands"]:
            raise PolicyError("tool 'shell' is declared but no commands are allowed")
        if "python_eval" in tools and not execution["python_enabled"]:
            raise PolicyError("tool 'python_eval' is declared but execution.python_enabled is false")
        if "web_request" in tools and not network["allowed_domains"]:
            raise PolicyError("tool 'web_request' is declared but no domains are allowed")

        triggers: list[Trigger] = []
        seen: set[str] = set()
        for item in data["escalation"]["triggers"]:
            if item["id"] in seen:
                raise PolicyError(f"duplicate escalation trigger {item['id']!r}")
            seen.add(item["id"])
            unknown = set(item["tools"]) - set(tools) - {"*"}
            if unknown:
                raise PolicyError(
                    f"trigger {item['id']!r} references undeclared tools: {sorted(unknown)}"
                )
            compiled = []
            for param, pattern in sorted(item.get("params", {}).items()):
                try:
                    compiled.append((param, re.compile(pattern)))
                except re.error as exc:
                    raise PolicyError(
                        f"trigger {item['id']!r} has an invalid pattern for {param!r}: {exc}"
                    ) from exc
            triggers.append(
                Trigger(
                    id=item["id"],
                    description=item["description"],
                    severity=item["severity"],
                    tools=frozenset(item["tools"]),
                    params=tuple(compiled),
                )
            )

        return cls(
            tools=tools,
            triggers=tuple(triggers),
            sandbox=sandbox,
            max_actions_per_session=data.get("budget", {}).get("max_actions_per_session"),
            raw=data,
        )

    def evaluate(self, tool: str, params: Any) -> Decision:
        """Decide whether a proposed call is allowed, denied, or needs a human."""
        rule = self.tools.get(tool)
        if rule is None:
            return Decision(Verdict.DENY, tool, "tool is not declared by the policy")
        if not isinstance(params, Mapping):
            return Decision(Verdict.DENY, tool, "tool parameters must be an object", rule.scope)
        if rule.requires_approval:
            return Decision(
                Verdict.ESCALATE, tool, "tool requires approval", rule.scope, severity="high"
            )
        for trigger in self.triggers:
            if trigger.matches(tool, params):
                return Decision(
                    Verdict.ESCALATE,
                    tool,
                    trigger.description,
                    rule.scope,
                    trigger_id=trigger.id,
                    severity=trigger.severity,
                )
        return Decision(Verdict.ALLOW, tool, "allowed by policy", rule.scope)

    def build_executor(
        self, workspace: str | Path, *, require_strong_isolation: bool = True
    ) -> ToolExecutor:
        """Create the reference executor for the built-in tools this policy declares."""
        network = self.sandbox["network"]
        execution = self.sandbox["execution"]
        isolation = self.sandbox["isolation"]
        limits = self.sandbox["resource_limits"]
        return ToolExecutor(
            workspace=workspace,
            allowed_tools=set(self.tools) & BUILTIN_TOOLS,
            allowed_commands=execution["allowed_commands"],
            allowed_domains=network["allowed_domains"],
            allow_http=bool(network.get("allow_http", False)),
            shell_enabled=execution["shell_enabled"],
            python_enabled=execution["python_enabled"],
            network_post_enabled=bool(network.get("allow_post", False)),
            timeout_seconds=limits["max_cpu_seconds_per_action"],
            max_memory_mb=limits["max_memory_mb"],
            process_isolation=isolation["type"],
            oci_runtime=isolation.get("runtime", "docker"),
            oci_image=isolation.get("image", ""),
            require_strong_isolation=require_strong_isolation,
        )


_IMAGE_BY_DIGEST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}$")

Approver = Callable[[Decision, Mapping[str, Any]], bool]
ToolCallable = Callable[[str, dict[str, Any]], dict[str, Any]]


class PolicyEnforcer:
    """Apply a policy before every effect. Fails closed on escalation without an approver."""

    def __init__(
        self,
        policy: Policy,
        execute: ToolCallable,
        approver: Approver | None = None,
    ):
        self.policy = policy
        self._execute = execute
        self._approver = approver
        self._lock = threading.Lock()
        self.actions_taken = 0
        self.audit: deque[dict[str, Any]] = deque(maxlen=MAX_AUDIT_ENTRIES)

    def execute(self, tool: str, params: Any) -> dict[str, Any]:
        decision = self.policy.evaluate(tool, params)
        if decision.verdict is Verdict.DENY:
            return self._refuse(decision, decision.reason)
        if decision.verdict is Verdict.ESCALATE:
            if self._approver is None:
                return self._refuse(decision, "approval required but no approver is configured")
            try:
                approved = self._approver(decision, params) is True
            except Exception:
                log.exception("approver_failed", tool=tool)
                approved = False
            if not approved:
                return self._refuse(decision, "approval was not granted")
        with self._lock:
            limit = self.policy.max_actions_per_session
            if limit is not None and self.actions_taken >= limit:
                return self._refuse(decision, "session action budget is exhausted")
            self.actions_taken += 1
        result = self._execute(tool, dict(params))
        result["decision"] = decision.to_dict()
        self._record(decision, executed=True, success=bool(result.get("success")))
        return result

    def _refuse(self, decision: Decision, error: str) -> dict[str, Any]:
        self._record(decision, executed=False, success=False, error=error)
        return {"success": False, "error": error, "decision": decision.to_dict()}

    def _record(self, decision: Decision, **fields: Any) -> None:
        entry = {**decision.to_dict(), **fields}
        self.audit.append(entry)
        log.info("policy_decision", **entry)
