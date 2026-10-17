"""Tests for the multimodal perception pipeline."""

from runtime.perception import PerceptionPipeline


class TestPerceptionPipeline:
    def test_text_perception(self):
        pp = PerceptionPipeline()
        percepts = pp.perceive("Hello, world!")
        assert len(percepts) == 1
        assert percepts[0].modality == "text"
        assert percepts[0].content == "Hello, world!"

    def test_language_detection_english(self):
        pp = PerceptionPipeline()
        percepts = pp.perceive("This is a test sentence in English")
        assert percepts[0].metadata.get("language") == "en"

    def test_dict_perception(self):
        pp = PerceptionPipeline()
        percepts = pp.perceive({"key": "value", "count": 42})
        assert len(percepts) == 1
        assert percepts[0].modality == "structured"

    def test_empty_input(self):
        pp = PerceptionPipeline()
        percepts = pp.perceive("")
        assert len(percepts) == 1
        assert percepts[0].modality == "text"
