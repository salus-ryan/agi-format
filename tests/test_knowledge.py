"""Tests for the semantic knowledge graph."""

from runtime.knowledge import KnowledgeGraph, Fact


class TestFact:
    def test_matches_exact(self):
        f = Fact("python", "is_a", "programming language")
        assert f.matches("python", "is_a", "programming language")
        assert not f.matches("java", "is_a", "programming language")

    def test_matches_wildcard(self):
        f = Fact("python", "is_a", "programming language")
        assert f.matches(subject="python")
        assert f.matches(predicate="is_a")
        assert f.matches(obj="programming language")
        assert f.matches()  # all wildcards

    def test_case_insensitive(self):
        f = Fact("Python", "IS_A", "Programming Language")
        assert f.matches("python", "is_a", "programming language")


class TestKnowledgeGraph:
    def test_assert_and_query(self):
        kg = KnowledgeGraph()
        kg.assert_fact("python", "is_a", "programming language")
        results = kg.query(subject="python")
        assert len(results) == 1
        assert results[0].object == "programming language"

    def test_reinforce_existing(self):
        kg = KnowledgeGraph()
        f1 = kg.assert_fact("python", "is_a", "language", confidence=0.5)
        initial_conf = f1.confidence
        f2 = kg.assert_fact("python", "is_a", "language", confidence=0.5)
        assert f2.confidence > initial_conf
        assert len(kg.facts) == 1  # no duplicate

    def test_conflict_detection(self):
        kg = KnowledgeGraph()
        f1 = kg.assert_fact("python", "is_a", "language", confidence=0.8)
        f2 = kg.assert_fact("python", "is_a", "snake", confidence=0.8)
        # Conflicting fact should reduce old confidence
        results = kg.query(subject="python", predicate="is_a")
        assert len(results) == 2
        language_fact = [f for f in results if f.object == "language"][0]
        assert language_fact.confidence < 0.8

    def test_retract(self):
        kg = KnowledgeGraph()
        kg.assert_fact("x", "y", "z")
        assert len(kg) == 1
        assert kg.retract("x", "y", "z")
        assert len(kg) == 0
        assert not kg.retract("x", "y", "z")  # already gone

    def test_query_text(self):
        kg = KnowledgeGraph()
        kg.assert_fact("python", "is_a", "programming language")
        kg.assert_fact("modal", "is_a", "serverless platform")
        kg.assert_fact("qwen", "runs_on", "gpu")
        results = kg.query_text("what programming language")
        assert any(f.subject == "python" for f in results)

    def test_get_context(self):
        kg = KnowledgeGraph()
        kg.assert_fact("python", "is_a", "language")
        ctx = kg.get_context("python")
        assert "Known Facts" in ctx
        assert "python" in ctx

    def test_decay(self):
        kg = KnowledgeGraph()
        f = kg.assert_fact("old", "fact", "here", confidence=0.15)
        # Manually age it
        f.timestamp = f.timestamp - 3600 * 100  # 100 hours ago
        pruned = kg.decay()
        assert pruned == 1
        assert len(kg) == 0

    def test_max_facts_pruning(self):
        kg = KnowledgeGraph()
        kg.MAX_FACTS = 5
        for i in range(10):
            kg.assert_fact(f"sub{i}", "pred", "obj", confidence=i * 0.1)
        assert len(kg) <= 5

    def test_summary(self):
        kg = KnowledgeGraph()
        kg.assert_fact("a", "b", "c")
        kg.assert_fact("d", "e", "f")
        s = kg.summary()
        assert s["total_facts"] == 2
        assert s["unique_subjects"] == 2
