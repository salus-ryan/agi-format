"""End-to-end integration tests.

These test the full pipeline without an LLM (mock the LLM client).
"""

import asyncio
import json

import pytest

from runtime.geolocation import GeoLocation
from runtime.knowledge import KnowledgeGraph
from runtime.learning import ReplayBuffer, Experience, OnlineLearner, Consolidator
from runtime.live_modules import (
    ToolExecutor, WorkingMemory, UserModel, ProceduralMemory,
    GoalStack, VectorMemory,
)
from runtime.perception import PerceptionPipeline
from runtime.planner import PlanSearch, Plan, PlanResult
from runtime.verification import FormalVerifier
from runtime.world_model import WorldModel, DynamicsModel


class TestFullPipeline:
    """Test the complete agent pipeline from input to output."""

    def test_perceive_ground_remember(self):
        """Test: input -> perception -> world model -> memory."""
        # Perceive
        pp = PerceptionPipeline()
        percepts = pp.perceive("Python is a great programming language")
        assert percepts[0].modality == "text"

        # Ground into world model
        geo = GeoLocation.from_modal_env(
            model_name="test", gpu="L4", owner_org="test", owner_workspace="test",
        )
        wm = WorldModel(geo)
        wm.observe("Python is great", [
            {"type": "concept", "name": "Python", "properties": {"kind": "language"}},
        ])
        assert any(n.name == "Python" for n in wm.nodes.values())

        # Remember
        vm = VectorMemory()
        vm.add("Python is a great programming language", importance=0.8)
        results = vm.search("programming language")
        assert len(results) > 0

        # Knowledge graph
        kg = KnowledgeGraph()
        kg.assert_fact("python", "is_a", "programming language")
        facts = kg.query(subject="python")
        assert len(facts) == 1

    def test_plan_execute_learn(self):
        """Test: plan -> tool execution -> dynamics update -> experience."""
        # Plan
        plan = Plan(
            description="List files in workspace",
            steps=[{"tool": "shell", "params": {"command": "ls /tmp/agent-workspace"}}],
            predicted_outcome="file listing", predicted_reward=0.7,
            risk="dir might be empty", plan_id="test-1",
        )

        # Execute
        te = ToolExecutor()
        result = te.execute("shell", {"command": "ls /tmp/agent-workspace"})
        assert result["success"]

        # Update dynamics
        dm = DynamicsModel()
        dm.record_transition("shell", "shell", "list files", result["result"], 0.8, True)
        pred = dm.predict_outcome("shell", "shell")
        assert pred["sample_count"] == 1
        assert pred["success_probability"] == 1.0

        # Store experience
        rb = ReplayBuffer(capacity=100)
        exp = Experience("list files", "shell", {"command": "ls"}, result["result"], 0.8)
        rb.add(exp)
        assert rb.size == 1

    def test_working_memory_user_model_goals(self):
        """Test: working memory + theory of mind + goal tracking."""
        wm = WorkingMemory()
        wm.attend("user wants to build AGI", source="user_input")
        wm.attend("Qwen3-8B available", source="world_model")

        um = UserModel()
        um.update_from_llm_analysis(json.dumps({
            "goals": ["build AGI agent"],
            "knowledge": ["Python", "ML"],
            "preferences": ["no stubs"],
        }))

        gs = GoalStack()
        main = gs.add_goal("build AGI agent", priority=0.9)
        sub1 = gs.add_goal("implement tools", priority=0.8, parent_id=main)
        gs.update_progress(sub1, 1.0, "done")

        # All should produce context
        assert "Working Memory" in wm.get_context()
        assert "User goals" in um.get_context()
        assert "Active Goals" in gs.get_context()

    def test_verification_pipeline(self):
        """Test: code in response -> formal verification."""
        fv = FormalVerifier()

        # Good code
        results = fv.verify_code("def add(x, y):\n    return x + y")
        assert all(r.verified for r in results)

        # Bad code
        results = fv.verify_code("def broken(:\n    pass")
        syntax_checks = [r for r in results if r.backend == "ast_parse"]
        assert not syntax_checks[0].verified

    def test_learning_cycle(self):
        """Test: experience -> replay buffer -> learning step -> consolidation."""
        rb = ReplayBuffer(capacity=1000)
        learner = OnlineLearner(rb)
        consolidator = Consolidator(rb)

        # Accumulate experiences
        for i in range(25):
            exp = Experience(f"step {i}", "shell", {}, f"result {i}", 0.5 + i * 0.02)
            learner.record_experience(exp)

        # Learning step
        assert learner.should_learn()
        metrics = learner.learn_step()
        assert "loss" in metrics

        # Track domain performance
        for i in range(10):
            consolidator.record_performance("coding", 0.6 + i * 0.03)

        # Consolidate
        result = consolidator.consolidate()
        assert isinstance(result, dict)

    def test_planner_reward_tracking(self):
        """Test: plan search history and keep/discard tracking."""
        ps = PlanSearch(n_candidates=3)

        # Simulate kept plans
        for i in range(5):
            plan = Plan(f"good plan {i}", [], "", 0.8, "", f"g{i}")
            result = PlanResult(plan, "worked", 0.8, True, [], 1.0)
            ps.plan_history.append(result)
            ps.plans_kept += 1

        # Simulate discarded plans
        for i in range(3):
            plan = Plan(f"bad plan {i}", [], "", 0.3, "", f"b{i}")
            result = PlanResult(plan, "failed", 0.2, False, [], 1.0)
            ps.plan_history.append(result)
            ps.plans_discarded += 1

        # History should contain both
        history = ps._format_history()
        assert "WORKED" in history
        assert "FAILED" in history

        # Average should be weighted
        avg = ps._recent_average_reward()
        assert 0.3 < avg < 0.8

    def test_dynamics_feeds_into_prediction(self):
        """Test: dynamics model improves world model predictions."""
        geo = GeoLocation.from_modal_env(
            model_name="test", gpu="L4", owner_org="test", owner_workspace="test",
        )
        wm = WorldModel(geo)

        # Before: no dynamics data
        pred_before = wm.predict("run python code", tool="python_eval", domain="coding")
        dynamics_before = [p for p in pred_before if p.get("source") == "learned_dynamics"]

        # Record transitions
        for i in range(20):
            wm.dynamics.record_transition("coding", "python_eval", f"s{i}", f"ok{i}", 0.9, True)

        # After: should have dynamics prediction
        pred_after = wm.predict("run python code", tool="python_eval", domain="coding")
        dynamics_after = [p for p in pred_after if p.get("source") == "learned_dynamics"]
        assert len(dynamics_after) > 0
        assert dynamics_after[0]["expected_reward"] > 0.8

    def test_knowledge_graph_conflict_resolution(self):
        """Test: conflicting facts reduce confidence of old facts."""
        kg = KnowledgeGraph()

        # Assert initial fact
        kg.assert_fact("earth", "shape", "flat", confidence=0.8)
        # Assert contradicting fact
        kg.assert_fact("earth", "shape", "round", confidence=0.9)

        facts = kg.query(subject="earth", predicate="shape")
        assert len(facts) == 2

        # The "flat" fact should have reduced confidence
        flat_fact = next(f for f in facts if f.object == "flat")
        assert flat_fact.confidence < 0.8

    def test_procedural_memory_accumulation(self):
        """Test: successful procedures stored and retrieved."""
        pm = ProceduralMemory()

        # Store a few skills
        pm.store_skill("sort_list", "coding", "Sort a list",
                        "1. Use sorted() for simple cases\n2. Use custom key for complex")
        pm.store_skill("http_get", "web", "Make HTTP GET",
                        "1. Use httpx.get(url)\n2. Check status code")

        # Reinforce sort_list
        pm.store_skill("sort_list", "coding", "Sort a list",
                        "Use sorted() with key parameter")
        assert pm.skills[0]["success_count"] == 2

        # Find by domain
        coding_skills = pm.find_skills("sort something", domain="coding")
        assert len(coding_skills) >= 1
        assert coding_skills[0]["name"] == "sort_list"

        # Context generation
        ctx = pm.get_context("how to sort", domain="coding")
        assert "sort_list" in ctx
