"""Tests for the world model and dynamics model."""

from runtime.geolocation import GeoLocation
from runtime.world_model import (
    WorldModel, DynamicsModel, Node, Edge,
    CAUSES, ENABLES, CONSTRAINS, LOCATED_IN, RUNS_ON,
)


def _make_geo():
    return GeoLocation.from_modal_env(
        model_name="test-model", gpu="L4",
        owner_org="test", owner_workspace="test",
        available_tools=["shell", "communicate"],
    )


class TestWorldModel:
    def test_seeds_from_geolocation(self):
        wm = WorldModel(_make_geo())
        assert len(wm.nodes) > 5
        assert len(wm.edges) > 5
        assert "self" in wm.nodes
        assert wm.nodes["self"].type == "agent"

    def test_observe_adds_nodes(self):
        wm = WorldModel(_make_geo())
        initial = len(wm.nodes)
        wm.observe("the sky is blue", [
            {"type": "concept", "name": "sky"},
            {"type": "concept", "name": "blue"},
        ])
        assert len(wm.nodes) > initial

    def test_predict_returns_predictions(self):
        wm = WorldModel(_make_geo())
        preds = wm.predict("running a shell command")
        # Should at least have regulation constraints
        assert len(preds) > 0
        assert any(p["type"] == "constraint" for p in preds)

    def test_predict_includes_dynamics(self):
        wm = WorldModel(_make_geo())
        # Record some transitions
        for i in range(10):
            wm.dynamics.record_transition("shell", "shell", "run cmd", "ok", 0.8, True)
        preds = wm.predict("running a shell command", tool="shell", domain="shell")
        dynamics_preds = [p for p in preds if p.get("source") == "learned_dynamics"]
        assert len(dynamics_preds) > 0
        assert dynamics_preds[0]["expected_reward"] > 0.5

    def test_explain_traces_causes(self):
        wm = WorldModel(_make_geo())
        explanations = wm.explain("test-model")
        # Should find something related to the agent
        assert isinstance(explanations, list)

    def test_get_context_for_reasoning(self):
        wm = WorldModel(_make_geo())
        ctx = wm.get_context_for_reasoning("what GPU am I running on")
        assert "L4" in ctx or "World Model" in ctx

    def test_summary(self):
        wm = WorldModel(_make_geo())
        s = wm.summary()
        assert s["total_nodes"] > 0
        assert s["total_edges"] > 0
        assert "node_types" in s
        assert "dynamics" in s


class TestDynamicsModel:
    def test_empty_prediction(self):
        dm = DynamicsModel()
        pred = dm.predict_outcome("coding", "python_eval")
        assert pred["sample_count"] == 0
        assert pred["confidence"] == 0.0
        assert pred["expected_reward"] == 0.5

    def test_record_and_predict(self):
        dm = DynamicsModel()
        for i in range(20):
            dm.record_transition("coding", "python_eval", f"state {i}",
                                 f"result {i}", 0.7, True)
        pred = dm.predict_outcome("coding", "python_eval")
        assert pred["sample_count"] == 20
        assert pred["expected_reward"] == 0.7
        assert pred["success_probability"] == 1.0
        assert pred["confidence"] > 0

    def test_variance_computation(self):
        dm = DynamicsModel()
        # All same reward -> zero variance
        for i in range(10):
            dm.record_transition("test", "tool", f"s{i}", f"r{i}", 0.5, True)
        pred = dm.predict_outcome("test", "tool")
        assert pred["reward_variance"] < 0.001

        # Mixed rewards -> nonzero variance
        dm2 = DynamicsModel()
        for i in range(10):
            dm2.record_transition("test", "tool", f"s{i}", f"r{i}",
                                  0.0 if i % 2 == 0 else 1.0, True)
        pred2 = dm2.predict_outcome("test", "tool")
        assert pred2["reward_variance"] > 0.2

    def test_summary(self):
        dm = DynamicsModel()
        dm.record_transition("a", "b", "s", "r", 0.5, True)
        s = dm.summary()
        assert s["total_transitions"] == 1
        assert s["known_pairs"] == 1
