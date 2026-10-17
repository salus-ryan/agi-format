"""Tests for the learning module: replay buffer, online learner, consolidator."""

from runtime.learning import ReplayBuffer, Experience, OnlineLearner, Consolidator


class TestExperience:
    def test_priority_computation(self):
        exp = Experience("state", "tool", {}, "obs", reward=0.8, novelty=0.5)
        assert exp.priority > 0

    def test_to_dict(self):
        exp = Experience("state", "tool", {"k": "v"}, "obs", 0.5)
        d = exp.to_dict()
        assert d["state_summary"] == "state"
        assert d["action_tool"] == "tool"
        assert d["reward"] == 0.5


class TestReplayBuffer:
    def test_add_and_size(self):
        rb = ReplayBuffer(capacity=100)
        for i in range(10):
            rb.add(Experience(f"s{i}", "t", {}, f"o{i}", 0.5))
        assert rb.size == 10
        assert rb.total_added == 10

    def test_capacity_limit(self):
        rb = ReplayBuffer(capacity=5)
        for i in range(10):
            rb.add(Experience(f"s{i}", "t", {}, f"o{i}", 0.5))
        assert rb.size == 5

    def test_sample(self):
        rb = ReplayBuffer(capacity=100)
        for i in range(20):
            rb.add(Experience(f"s{i}", "t", {}, f"o{i}", 0.5 + i * 0.01))
        batch, weights = rb.sample(5)
        assert len(batch) == 5
        assert len(weights) == 5
        assert all(w > 0 for w in weights)

    def test_sample_larger_than_buffer(self):
        rb = ReplayBuffer(capacity=100)
        for i in range(3):
            rb.add(Experience(f"s{i}", "t", {}, f"o{i}", 0.5))
        batch, weights = rb.sample(10)
        assert len(batch) == 3


class TestOnlineLearner:
    def test_record_and_learn(self):
        rb = ReplayBuffer(capacity=100)
        learner = OnlineLearner(rb)
        for i in range(20):
            exp = Experience(f"s{i}", "t", {}, f"o{i}", 0.5 + i * 0.01)
            learner.record_experience(exp)
        assert rb.size == 20
        # Should be ready to learn
        assert learner.should_learn()
        metrics = learner.learn_step()
        assert "loss" in metrics

    def test_summary(self):
        rb = ReplayBuffer(capacity=100)
        learner = OnlineLearner(rb)
        s = learner.summary()
        assert "updates_performed" in s


class TestConsolidator:
    def test_record_performance(self):
        rb = ReplayBuffer(capacity=100)
        con = Consolidator(rb)
        for i in range(10):
            con.record_performance("coding", 0.5 + i * 0.05)
        assert "coding" in con.domain_performance

    def test_consolidate_empty(self):
        rb = ReplayBuffer(capacity=100)
        con = Consolidator(rb)
        result = con.consolidate()
        assert isinstance(result, dict)

    def test_summary(self):
        rb = ReplayBuffer(capacity=100)
        con = Consolidator(rb)
        s = con.summary()
        assert "monitored_domains" in s
