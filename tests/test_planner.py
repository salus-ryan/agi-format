"""Tests for the plan search module."""

import asyncio

from runtime.planner import PlanSearch, Plan, PlanResult


class TestPlan:
    def test_to_dict(self):
        p = Plan(
            description="test plan", steps=[{"tool": "shell", "params": {"command": "ls"}}],
            predicted_outcome="list files", predicted_reward=0.7,
            risk="might be empty", plan_id="p1",
        )
        d = p.to_dict()
        assert d["description"] == "test plan"
        assert d["predicted_reward"] == 0.7
        assert len(d["steps"]) == 1


class TestPlanSearch:
    def test_compute_plan_reward_all_success(self):
        ps = PlanSearch()
        plan = Plan("test", [{"tool": "shell"}, {"tool": "file_read"}], "", 0.5, "", "p1")
        reward = ps._compute_plan_reward(plan, [
            {"success": True, "result": "ok"},
            {"success": True, "result": "ok"},
        ])
        assert reward == 1.0

    def test_compute_plan_reward_partial_failure(self):
        ps = PlanSearch()
        plan = Plan("test", [{"tool": "shell"}, {"tool": "file_read"}], "", 0.5, "", "p1")
        reward = ps._compute_plan_reward(plan, [
            {"success": True, "result": "ok"},
            {"success": False, "result": "fail"},
        ])
        assert 0 < reward < 1.0

    def test_compute_plan_reward_no_tools(self):
        ps = PlanSearch()
        plan = Plan("test", [], "", 0.5, "", "p1")
        reward = ps._compute_plan_reward(plan, [])
        assert reward == 0.3

    def test_history_tracking(self):
        ps = PlanSearch()
        plan = Plan("test plan", [], "", 0.5, "none", "p1")
        result = PlanResult(plan, "ok", 0.8, True, [], 1.0)
        ps.plan_history.append(result)
        ps.plans_kept += 1

        history = ps._format_history()
        assert "WORKED" in history
        assert "test plan" in history

    def test_recent_average_reward(self):
        ps = PlanSearch()
        for i in range(5):
            plan = Plan(f"p{i}", [], "", 0.5, "", f"p{i}")
            result = PlanResult(plan, "ok", 0.6, True, [], 1.0)
            ps.plan_history.append(result)
        assert ps._recent_average_reward() == 0.6

    def test_summary(self):
        ps = PlanSearch(n_candidates=3)
        s = ps.summary()
        assert s["plans_generated"] == 0
        assert s["plans_kept"] == 0
