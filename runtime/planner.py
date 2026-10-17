"""Plan search: generate candidates, simulate, select, execute, evaluate, keep/discard.

Adapted from Karpathy's autoresearch pattern:
  1. Generate N candidate plans using the LLM
  2. Score each plan against the world model (predict outcomes)
  3. Execute the best plan
  4. Measure the actual outcome
  5. Keep if improved, discard if not
  6. Feed results back for next planning cycle

The LLM is the heuristic function -- it proposes what to try based on the
full history of what worked and what didn't. The "tree" is implicit in the
agent's reasoning about what to explore next.
"""

import json
import re
import time
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass
class Plan:
    """A candidate plan with predicted outcome."""
    description: str
    steps: list[dict]          # [{"tool": "...", "params": {...}, "rationale": "..."}]
    predicted_outcome: str
    predicted_reward: float    # LLM's estimate of how well this will work [0, 1]
    risk: str                  # what could go wrong
    plan_id: str = ""

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "description": self.description,
            "steps": self.steps,
            "predicted_outcome": self.predicted_outcome,
            "predicted_reward": self.predicted_reward,
            "risk": self.risk,
        }


@dataclass
class PlanResult:
    """Outcome of an executed plan."""
    plan: Plan
    actual_outcome: str
    actual_reward: float
    kept: bool                 # was this plan kept (improved things)?
    tool_results: list[dict]
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "actual_outcome": self.actual_outcome[:300],
            "actual_reward": self.actual_reward,
            "kept": self.kept,
            "tools_executed": len(self.tool_results),
            "elapsed_seconds": self.elapsed_seconds,
        }


class PlanSearch:
    """Tree search over candidate plans using the LLM as the heuristic.

    The search is breadth-first at depth 1 (generate N candidates, pick best)
    but depth-first in execution (multi-step plans execute sequentially with
    checkpoints so we can revert on failure).

    History of past plans (kept and discarded) is fed back to the LLM so it
    learns what works and what doesn't, even within a single session.
    """

    def __init__(self, n_candidates: int = 3, max_plan_steps: int = 5):
        self.n_candidates = n_candidates
        self.max_plan_steps = max_plan_steps
        self.plan_history: list[PlanResult] = []
        self.plans_generated = 0
        self.plans_kept = 0
        self.plans_discarded = 0

    async def search(self, query: str, context: dict, llm_fn, tool_fn,
                     world_model=None, escalation_fn=None) -> PlanResult | None:
        """Full plan search cycle.

        Args:
            query: the user's request / current objective
            context: dict with keys like 'working_memory', 'user_model',
                     'episodic_memories', 'world_state'
            llm_fn: async fn(messages, max_tokens, temperature) -> str
            tool_fn: fn(tool, params) -> dict (tool executor)
            world_model: optional WorldModel for simulation
            escalation_fn: optional fn(tool, params) -> EscalationDecision

        Returns:
            PlanResult if a plan was executed, None if task is simple enough
            to skip planning.
        """
        # Step 1: should we even plan? Simple queries don't need it.
        needs_planning = await self._needs_planning(query, context, llm_fn)
        if not needs_planning:
            return None

        start = time.time()

        # Step 2: generate N candidate plans
        candidates = await self._generate_candidates(query, context, llm_fn)
        if not candidates:
            return None

        # Step 3: score candidates against world model
        if world_model:
            candidates = await self._simulate_plans(candidates, context, llm_fn, world_model)

        # Step 4: rank and select best
        candidates.sort(key=lambda p: p.predicted_reward, reverse=True)
        selected = candidates[0]

        log.info("plan_selected",
                 plan_id=selected.plan_id,
                 description=selected.description[:100],
                 predicted_reward=selected.predicted_reward,
                 candidates_considered=len(candidates))

        # Step 5: execute the plan (with escalation checks)
        tool_results = []
        for step in selected.steps:
            tool = step.get("tool", "communicate")
            params = step.get("params", {})

            if tool == "communicate":
                tool_results.append({
                    "tool": "communicate", "params": params,
                    "success": True, "result": params.get("message", "")[:200],
                })
                continue

            # Check escalation before executing
            if escalation_fn:
                escalation = escalation_fn(tool, params)
                if escalation.should_escalate:
                    tool_results.append({
                        "tool": tool, "params": params,
                        "success": False,
                        "result": f"ESCALATION: {escalation.why} -- {escalation.what}",
                    })
                    log.info("plan_step_escalated", tool=tool,
                             why=escalation.why[:200])
                    break

            result = tool_fn(tool, params)
            tool_results.append({
                "tool": tool, "params": params,
                "success": result.get("success"),
                "result": str(result.get("result", result.get("error", "")))[:500],
            })

            # If a step fails, stop executing
            if not result.get("success"):
                log.info("plan_step_failed", tool=tool,
                         error=result.get("error", "")[:200])
                break

        # Step 6: evaluate the actual outcome
        actual_reward = self._compute_plan_reward(selected, tool_results)
        actual_outcome = self._summarize_results(tool_results)

        # Step 7: keep or discard
        # Keep if: reward > 0.3 (absolute threshold) or better than average of recent plans
        recent_avg = self._recent_average_reward()
        kept = actual_reward > max(0.3, recent_avg * 0.9)

        elapsed = time.time() - start
        plan_result = PlanResult(
            plan=selected,
            actual_outcome=actual_outcome,
            actual_reward=actual_reward,
            kept=kept,
            tool_results=tool_results,
            elapsed_seconds=round(elapsed, 2),
        )

        self.plan_history.append(plan_result)
        if kept:
            self.plans_kept += 1
        else:
            self.plans_discarded += 1

        log.info("plan_evaluated",
                 kept=kept, actual_reward=actual_reward,
                 predicted_reward=selected.predicted_reward,
                 elapsed=elapsed)

        return plan_result

    async def _needs_planning(self, query: str, context: dict, llm_fn) -> bool:
        """Ask the LLM if this query needs multi-step planning or is simple enough
        to answer directly."""
        try:
            response = await llm_fn([
                {"role": "system", "content": (
                    "Determine if this request needs multi-step planning with tool use, "
                    "or if it can be answered directly from knowledge. "
                    "Reply with ONLY 'plan' or 'direct'. Nothing else."
                )},
                {"role": "user", "content": query[:500]},
            ], max_tokens=8, temperature=0.1)
            return "plan" in response.lower()
        except Exception:
            return False

    async def _generate_candidates(self, query: str, context: dict,
                                   llm_fn) -> list[Plan]:
        """Generate N candidate plans using the LLM."""
        self.plans_generated += self.n_candidates

        # Build history context so the LLM knows what's been tried
        history_context = self._format_history()

        try:
            response = await llm_fn([
                {"role": "system", "content": (
                    f"Generate exactly {self.n_candidates} candidate plans to accomplish "
                    "the user's request. Each plan should have a different approach.\n\n"
                    "Available tools:\n"
                    "- shell: {\"command\": \"...\"}\n"
                    "- file_read: {\"path\": \"...\"}\n"
                    "- file_write: {\"path\": \"/tmp/agent-workspace/...\", \"content\": \"...\"}\n"
                    "- web_request: {\"url\": \"...\", \"method\": \"GET|POST\"}\n"
                    "- python_eval: {\"code\": \"...\"}\n"
                    "- communicate: {\"message\": \"...\"}\n\n"
                    "Return JSON array of plans, each with:\n"
                    '{"description": "...", "steps": [{"tool": "...", "params": {...}, '
                    '"rationale": "..."}], "predicted_outcome": "...", '
                    '"predicted_reward": 0.0-1.0, "risk": "..."}\n\n'
                    "JSON only, no markdown fences."
                    + (f"\n\n{history_context}" if history_context else "")
                    + (f"\n\nCurrent context:\n{json.dumps(context, default=str)[:1000]}"
                       if context else "")
                )},
                {"role": "user", "content": query[:1000]},
            ], max_tokens=1500, temperature=0.7)

            # Parse the JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []

            plans_data = json.loads(json_match.group())
            plans = []
            for i, pd in enumerate(plans_data[:self.n_candidates]):
                if not isinstance(pd, dict):
                    continue
                plans.append(Plan(
                    description=pd.get("description", f"Plan {i+1}"),
                    steps=pd.get("steps", [])[:self.max_plan_steps],
                    predicted_outcome=pd.get("predicted_outcome", ""),
                    predicted_reward=float(pd.get("predicted_reward", 0.5)),
                    risk=pd.get("risk", "unknown"),
                    plan_id=f"plan-{self.plans_generated}-{i}",
                ))
            return plans

        except Exception as e:
            log.error("plan_generation_failed", error=str(e))
            return []

    async def _simulate_plans(self, plans: list[Plan], context: dict,
                              llm_fn, world_model) -> list[Plan]:
        """Simulate each plan using both dynamics model predictions and LLM reasoning."""
        world_summary = world_model.summary()

        for plan in plans:
            try:
                # Step 1: Use the dynamics model for data-driven predictions
                dynamics_predictions = []
                for step in plan.steps:
                    tool = step.get("tool", "communicate")
                    # Classify domain from context
                    domain = context.get("domain", "general") if context else "general"
                    prediction = world_model.dynamics.predict_outcome(
                        domain=domain, tool=tool,
                        state_summary=plan.description[:200],
                    )
                    dynamics_predictions.append({
                        "tool": tool,
                        "expected_reward": prediction.get("expected_reward", 0.5),
                        "success_probability": prediction.get("success_probability", 0.5),
                        "confidence": prediction.get("confidence", 0.0),
                    })

                # Step 2: Use causal graph predictions for entities in the plan
                causal_predictions = world_model.predict(plan.description[:200])

                # Step 3: Combine dynamics + causal + LLM reasoning
                dynamics_context = json.dumps(dynamics_predictions, default=str)[:400]
                causal_context = json.dumps(causal_predictions, default=str)[:300]

                response = await llm_fn([
                    {"role": "system", "content": (
                        "Given the world state, dynamics model predictions, and causal graph, "
                        "predict the outcome of this plan. "
                        "Return JSON: {\"adjusted_reward\": 0.0-1.0, \"failure_probability\": 0.0-1.0, "
                        "\"reasoning\": \"...\"}. JSON only."
                    )},
                    {"role": "user", "content": (
                        f"World state: {json.dumps(world_summary, default=str)[:400]}\n"
                        f"Dynamics predictions per step: {dynamics_context}\n"
                        f"Causal predictions: {causal_context}\n"
                        f"Plan: {json.dumps(plan.to_dict(), default=str)[:400]}"
                    )},
                ], max_tokens=256, temperature=0.2)

                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    sim = json.loads(json_match.group())
                    adjusted = float(sim.get("adjusted_reward", plan.predicted_reward))
                    failure_prob = float(sim.get("failure_probability", 0.3))

                    # Weight dynamics model confidence into the blend
                    high_confidence_steps = [d for d in dynamics_predictions
                                             if d["confidence"] > 0.5]
                    if high_confidence_steps:
                        # Dynamics model has real data -- weight it more heavily
                        dynamics_reward = sum(d["expected_reward"] * d["success_probability"]
                                             for d in high_confidence_steps) / len(high_confidence_steps)
                        plan.predicted_reward = 0.4 * dynamics_reward + 0.6 * adjusted * (1 - failure_prob)
                    else:
                        plan.predicted_reward = adjusted * (1 - failure_prob)

            except Exception:
                pass  # keep original prediction

        return plans

    def _compute_plan_reward(self, plan: Plan, tool_results: list[dict]) -> float:
        """Compute actual reward from execution results."""
        if not tool_results:
            return 0.3  # no tools executed, neutral

        successes = sum(1 for r in tool_results if r.get("success"))
        total = len(tool_results)
        success_rate = successes / total

        # Bonus for completing all steps
        completion = len(tool_results) / max(1, len(plan.steps))

        return 0.4 * success_rate + 0.4 * completion + 0.2 * (1.0 if successes == total else 0.0)

    def _summarize_results(self, tool_results: list[dict]) -> str:
        """Summarize tool execution results."""
        if not tool_results:
            return "No tools executed."
        parts = []
        for r in tool_results:
            status = "OK" if r.get("success") else "FAIL"
            parts.append(f"{r['tool']} [{status}]: {r.get('result', '')[:100]}")
        return " | ".join(parts)

    def _recent_average_reward(self, window: int = 10) -> float:
        """Average reward of recent plans."""
        recent = [p.actual_reward for p in self.plan_history[-window:]]
        return sum(recent) / len(recent) if recent else 0.3

    def _format_history(self) -> str:
        """Format plan history for the LLM to learn from."""
        if not self.plan_history:
            return ""

        recent = self.plan_history[-10:]
        lines = ["## Past plans (learn from these):"]

        kept = [p for p in recent if p.kept]
        discarded = [p for p in recent if not p.kept]

        if kept:
            lines.append("\nApproaches that WORKED:")
            for p in kept[-5:]:
                lines.append(f"- {p.plan.description[:100]} (reward={p.actual_reward:.2f})")

        if discarded:
            lines.append("\nApproaches that FAILED (don't repeat these):")
            for p in discarded[-5:]:
                lines.append(f"- {p.plan.description[:100]} (reward={p.actual_reward:.2f})")

        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "plans_generated": self.plans_generated,
            "plans_kept": self.plans_kept,
            "plans_discarded": self.plans_discarded,
            "recent_avg_reward": round(self._recent_average_reward(), 3),
            "history_length": len(self.plan_history),
        }
